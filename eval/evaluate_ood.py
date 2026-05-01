"""
Energy Score 기반 OOD 탐지 평가 스크립트

1. best.pt 로드 + YOLOv8 Detect head의 cv3에 hook 등록
2. ID 이미지(yolo_dataset/images/val) + OOD 이미지(ood_test_images.txt)에 대해
   raw logit 추출 → energy score 계산
3. AUROC, FPR@TPR95 계산 및 ROC 곡선 저장

사용 예시:
  # 전체 데이터
  python eval/evaluate_ood.py

  # 빠른 검증 (각 2000장)
  python eval/evaluate_ood.py --max-id 2000 --max-ood 2000

  # 모델/경로 직접 지정
  python eval/evaluate_ood.py --model runs/train/exp/weights/best.pt
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OOD_LIST     = DATA_DIR / "ood_test_images.txt"
ID_IMAGE_DIR = Path("D:/orchard_data/yolo_dataset/images/val")


# ─────────────────────────────────────────────────
# 인수 파싱
# ─────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Energy Score OOD 탐지 평가",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    default=str(PROJECT_ROOT / "best.pt"),
                   help="학습된 모델 경로 (.pt)")
    p.add_argument("--out-dir",  default=str(PROJECT_ROOT / "eval" / "results"),
                   help="결과 저장 디렉토리")
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--device",   default=None,
                   help="장치 (예: 0, cpu). 기본: 자동 감지")
    p.add_argument("--max-id",   type=int, default=None, metavar="N",
                   help="ID 이미지 최대 샘플 수 (None: 전체)")
    p.add_argument("--max-ood",  type=int, default=None, metavar="N",
                   help="OOD 이미지 최대 샘플 수 (None: 전체)")
    return p.parse_args()


# ─────────────────────────────────────────────────
# Logit Hook
# ─────────────────────────────────────────────────
class LogitHook:
    """YOLOv8 Detect head의 cv3에서 sigmoid 이전 raw logit을 캡처."""

    def __init__(self):
        self._buf:     list[torch.Tensor] = []
        self._handles: list               = []

    def register(self, yolo_model):
        from ultralytics.nn.modules import Detect
        detect = next(
            (m for m in yolo_model.modules() if isinstance(m, Detect)), None
        )
        if detect is None:
            raise RuntimeError("Detect head를 찾을 수 없음 — 모델 구조 확인 필요")
        for cv3 in detect.cv3:
            self._handles.append(cv3.register_forward_hook(self._capture))

    def _capture(self, module, inp, out: torch.Tensor):
        # out: [B, nc, H, W] — sigmoid 이전 raw logit
        B, nc, H, W = out.shape
        self._buf.append(out.detach().reshape(B, nc, H * W))

    def pop_energy(self) -> torch.Tensor:
        """버퍼에서 이미지당 energy score를 계산 후 초기화.

        낮을수록 In-Distribution, 높을수록 OOD.
        """
        if not self._buf:
            raise RuntimeError("hook 버퍼 비어 있음 — forward pass 실행 여부 확인")
        logits = torch.cat(self._buf, dim=-1)   # [B, nc, total_anchors]
        self._buf.clear()
        anchor_energy = -torch.logsumexp(logits, dim=1)  # [B, total_anchors]
        return anchor_energy.min(dim=-1).values           # [B] — 가장 ID에 가까운 앵커

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ─────────────────────────────────────────────────
# 이미지 경로 로드
# ─────────────────────────────────────────────────
def load_id_images(max_n: int | None) -> list[Path]:
    if not ID_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"ID 이미지 디렉토리 없음: {ID_IMAGE_DIR}\n"
            "  → data/build_yolo_dataset.py 를 먼저 실행하세요."
        )
    paths = sorted(ID_IMAGE_DIR.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"ID 이미지 없음: {ID_IMAGE_DIR}")
    if max_n:
        paths = paths[:max_n]
    return paths


def load_ood_images(max_n: int | None) -> list[Path]:
    if not OOD_LIST.exists():
        raise FileNotFoundError(
            f"OOD 목록 없음: {OOD_LIST}\n"
            "  → python eval/prepare_ood_testset.py 를 먼저 실행하세요."
        )
    lines = OOD_LIST.read_text(encoding="utf-8").splitlines()
    paths = [Path(l.strip()) for l in lines if l.strip()]
    if not paths:
        raise ValueError("ood_test_images.txt가 비어 있음")
    if max_n:
        paths = paths[:max_n]
    return paths


# ─────────────────────────────────────────────────
# 전처리 (letterbox + normalize)
# ─────────────────────────────────────────────────
def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w  = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def _load_batch(paths: list[Path], imgsz: int, device: str) -> torch.Tensor:
    tensors = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            img = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        img = _letterbox(img, imgsz)
        img = img[:, :, ::-1]                                  # BGR → RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1))    # HWC → CHW
        tensors.append(torch.from_numpy(img).float().div(255.0))
    # "0" → "cuda:0" 변환 (torch .to()는 "cuda:0" 형식 요구)
    torch_device = f"cuda:{device}" if device.isdigit() else device
    return torch.stack(tensors).to(torch_device)


# ─────────────────────────────────────────────────
# 에너지 스코어 계산
# ─────────────────────────────────────────────────
def compute_energies(
    model,
    hook: LogitHook,
    image_paths: list[Path],
    batch_size: int,
    imgsz: int,
    device: str,
    label: str,
) -> list[float]:
    torch_device = f"cuda:{device}" if device.isdigit() else device
    yolo_model   = model.model.to(torch_device)
    yolo_model.eval()
    energies: list[float] = []
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        img_tensor  = _load_batch(batch_paths, imgsz, device)
        hook._buf.clear()
        with torch.no_grad():
            yolo_model(img_tensor)
        energies.extend(hook.pop_energy().tolist())
        print(f"\r  {label}: {min(i + batch_size, total):,}/{total:,}", end="", flush=True)

    print()
    return energies


# ─────────────────────────────────────────────────
# AUROC / FPR@TPR95
# ─────────────────────────────────────────────────
def evaluate(
    id_scores: list[float],
    ood_scores: list[float],
    out_dir: Path,
) -> tuple[float, float]:
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        print("[ERROR] scikit-learn 미설치 → pip install scikit-learn")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.array([0] * len(id_scores) + [1] * len(ood_scores))
    scores = np.array(id_scores + ood_scores)

    auroc        = roc_auc_score(y_true, scores)
    fpr, tpr, _  = roc_curve(y_true, scores)

    idx       = np.searchsorted(tpr, 0.95)
    fpr_at_95 = float(fpr[min(idx, len(fpr) - 1)])

    print("\n── 평가 결과 ──────────────────────────────────")
    print(f"  AUROC       : {auroc:.4f}  (목표 ≥ 0.90)")
    print(f"  FPR@TPR95   : {fpr_at_95:.4f}  (낮을수록 좋음)")
    print(f"  ID 샘플     : {len(id_scores):,}")
    print(f"  OOD 샘플    : {len(ood_scores):,}")
    print("───────────────────────────────────────────────")

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "id_energies.npy",  np.array(id_scores,  dtype=np.float32))
    np.save(out_dir / "ood_energies.npy", np.array(ood_scores, dtype=np.float32))

    # ROC 곡선
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.axvline(fpr_at_95, color="r", linestyle=":", lw=1,
               label=f"FPR@TPR95 = {fpr_at_95:.4f}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Energy Score OOD Detection — ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    # 에너지 분포 히스토그램
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_scores,  bins=100, alpha=0.6, label="ID",  density=True)
    ax.hist(ood_scores, bins=100, alpha=0.6, label="OOD", density=True)
    ax.set_xlabel("Energy Score")
    ax.set_ylabel("Density")
    ax.set_title("Energy Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "energy_hist.png", dpi=150)
    plt.close()

    summary = (
        f"AUROC:       {auroc:.4f}\n"
        f"FPR@TPR95:   {fpr_at_95:.4f}\n"
        f"ID  samples: {len(id_scores)}\n"
        f"OOD samples: {len(ood_scores)}\n"
    )
    (out_dir / "results.txt").write_text(summary, encoding="utf-8")

    print(f"\n✓ 결과 저장: {out_dir}")
    print("  roc_curve.png  energy_hist.png  results.txt")
    print("  id_energies.npy  ood_energies.npy")

    return auroc, fpr_at_95


# ─────────────────────────────────────────────────
def main():
    args = parse_args()

    from ultralytics import YOLO
    print(f"모델 로드: {args.model}")
    model = YOLO(args.model)

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"장치: {device}")

    hook = LogitHook()
    hook.register(model.model)

    try:
        print("\nID 이미지 로드 중...")
        id_paths = load_id_images(args.max_id)
        print(f"  {len(id_paths):,}장")

        print("OOD 이미지 로드 중...")
        ood_paths = load_ood_images(args.max_ood)
        print(f"  {len(ood_paths):,}장")

        print(f"\n에너지 스코어 계산 중 (batch={args.batch}, imgsz={args.imgsz})...")
        id_energies  = compute_energies(model, hook, id_paths,  args.batch, args.imgsz, device, "ID ")
        ood_energies = compute_energies(model, hook, ood_paths, args.batch, args.imgsz, device, "OOD")

        evaluate(id_energies, ood_energies, Path(args.out_dir))

    finally:
        hook.remove()


if __name__ == "__main__":
    main()
