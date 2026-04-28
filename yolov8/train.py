"""
YOLOv8 학습 스크립트 — 과수원 OOD 탐지 프로젝트

다른 컴퓨터에서 실행할 경우:
  --dataset-root 로 yolo_dataset 경로만 지정하면 dataset.yaml을 자동 생성합니다.

사용 예시:
  # 기본 실행 (D:/orchard_data/yolo_dataset 사용)
  python yolov8/train.py

  # 다른 경로의 데이터셋 사용
  python yolov8/train.py --dataset-root E:/orchard_data/yolo_dataset

  # 모델·에폭 지정
  python yolov8/train.py --model yolov8l.pt --epochs 100 --imgsz 1280

  # 이전 학습 이어서
  python yolov8/train.py --resume runs/train/exp1/weights/last.pt
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import platform
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────
DEFAULT_DATASET_ROOT = Path(r"D:/orchard_data/yolo_dataset")
DEFAULT_DATA_YAML    = Path(__file__).parent.parent / "data" / "dataset.yaml"

CLASS_NAMES = ["사람", "손수레", "사다리"]


# ─────────────────────────────────────────────────────
# 인수 파싱
# ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv8 orchard OOD 기반 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", type=str, default=None,
        help="yolo_dataset 루트 경로. 지정 시 dataset.yaml을 자동 생성",
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="dataset.yaml 경로 (--dataset-root 보다 우선)",
    )
    p.add_argument(
        "--model", type=str, default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                 "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 모델 크기",
    )
    p.add_argument(
        "--epochs", type=int, default=100,
        help="학습 에폭 수 (목표 mAP≥95% 위해 최소 50 권장)",
    )
    p.add_argument(
        "--imgsz", type=int, default=1280,
        help="입력 이미지 크기 (원본 1920×1080 → 1280 권장)",
    )
    p.add_argument(
        "--batch", type=int, default=-1,
        help="배치 크기 (-1: GPU 메모리 기준 자동)",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="DataLoader 워커 수",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="학습 장치 (예: 0, 0,1, cpu). 기본값: 자동 감지",
    )
    p.add_argument(
        "--project", type=str, default="runs/train",
        help="결과 저장 상위 디렉토리",
    )
    p.add_argument(
        "--name", type=str, default=None,
        help="실험 이름 (기본값: 날짜_모델명)",
    )
    p.add_argument(
        "--resume", type=str, default=None, metavar="LAST_PT",
        help="last.pt 경로 지정 시 해당 체크포인트에서 재개",
    )
    p.add_argument(
        "--save-period", type=int, default=10,
        help="N 에폭마다 체크포인트 저장",
    )
    p.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience (0: 비활성화)",
    )
    p.add_argument(
        "--lr0", type=float, default=0.01,
        help="초기 학습률",
    )
    p.add_argument(
        "--cos-lr", action="store_true", default=True,
        help="Cosine LR 스케줄러 사용",
    )
    p.add_argument(
        "--cache", type=str, default="disk",
        choices=["ram", "disk", "false"],
        help="이미지 캐시 방식 (대용량 데이터셋: disk 권장)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────
# 환경 정보 출력
# ─────────────────────────────────────────────────────
def print_env_info() -> str:
    """GPU/CPU/Python 환경 요약 출력, 장치 문자열 반환."""
    print("\n" + "=" * 55)
    print("  환경 정보")
    print("=" * 55)
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  OS       : {platform.system()} {platform.release()}")

    try:
        import torch
        print(f"  PyTorch  : {torch.__version__}")
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            for i in range(n_gpu):
                prop = torch.cuda.get_device_properties(i)
                vram = prop.total_memory / 1024 ** 3
                print(f"  GPU [{i}]  : {prop.name}  ({vram:.1f} GB VRAM)")
            device = "0"
        else:
            print("  GPU      : 없음 — CPU로 학습 (매우 느림)")
            device = "cpu"
    except ImportError:
        print("  PyTorch  : 미설치")
        device = "cpu"

    try:
        import ultralytics
        print(f"  YOLOv8   : ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  YOLOv8   : 미설치 → pip install ultralytics")
        sys.exit(1)

    print("=" * 55 + "\n")
    return device


# ─────────────────────────────────────────────────────
# dataset.yaml 처리
# ─────────────────────────────────────────────────────
def resolve_data_yaml(args: argparse.Namespace) -> str:
    """
    --data 또는 --dataset-root 에서 사용할 yaml 경로 결정.
    --dataset-root 지정 시 임시 yaml 생성.
    """
    if args.data:
        path = Path(args.data)
        if not path.exists():
            raise FileNotFoundError(f"dataset.yaml 없음: {path}")
        print(f"dataset.yaml: {path}")
        return str(path)

    if args.dataset_root:
        root = Path(args.dataset_root)
    else:
        # 기본 yaml 파일이 있으면 사용
        if DEFAULT_DATA_YAML.exists():
            print(f"dataset.yaml: {DEFAULT_DATA_YAML}")
            return str(DEFAULT_DATA_YAML)
        root = DEFAULT_DATASET_ROOT

    # root 경로로 임시 yaml 생성
    if not root.exists():
        raise FileNotFoundError(
            f"yolo_dataset 경로 없음: {root}\n"
            "  → --dataset-root 로 올바른 경로를 지정하세요."
        )
    _validate_dataset_root(root)

    yaml_content = {
        "path":  root.as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    # 프로젝트 data/ 아래에 저장 (재사용 가능)
    out_yaml = Path(__file__).parent.parent / "data" / "dataset.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
    print(f"dataset.yaml 생성: {out_yaml}")
    return str(out_yaml)


def _validate_dataset_root(root: Path):
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "yolo_dataset 하위 디렉토리 없음:\n" +
            "\n".join(f"  {m}" for m in missing) +
            "\n  → data/build_yolo_dataset.py 를 먼저 실행하세요."
        )


# ─────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────
def train(args: argparse.Namespace, data_yaml: str, device: str):
    from ultralytics import YOLO

    # 실험 이름 자동 생성
    exp_name = args.name or f"{datetime.now().strftime('%Y%m%d_%H%M')}_{args.model.replace('.pt','')}"

    # 재개 모드
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume 파일 없음: {resume_path}")
        print(f"체크포인트 재개: {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
        return

    print(textwrap.dedent(f"""
    ── 학습 설정 ──────────────────────────────────────
      모델     : {args.model}
      에폭     : {args.epochs}
      이미지   : {args.imgsz}px
      배치     : {'auto' if args.batch == -1 else args.batch}
      장치     : {device}
      캐시     : {args.cache}
      실험명   : {exp_name}
      저장위치 : {args.project}/{exp_name}
    ───────────────────────────────────────────────────
    """))

    model = YOLO(args.model)

    cache_val = False if args.cache == "false" else args.cache

    model.train(
        data        = data_yaml,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        workers     = args.workers,
        device      = device,
        project     = args.project,
        name        = exp_name,
        save_period = args.save_period,
        patience    = args.patience,
        lr0         = args.lr0,
        cos_lr      = args.cos_lr,
        cache       = cache_val,
        # 정확도 우선 augmentation 설정
        degrees     = 5.0,       # 약한 회전 (과수원 환경)
        fliplr      = 0.5,
        mosaic      = 1.0,
        mixup       = 0.1,
        # 로깅
        plots       = True,
        verbose     = True,
    )

    # 결과 경로 안내
    result_dir = Path(args.project) / exp_name
    print(f"\n✓ 학습 완료: {result_dir}")
    print(f"  best.pt  : {result_dir / 'weights' / 'best.pt'}")
    print(f"  결과 그래프: {result_dir / 'results.png'}")


# ─────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = print_env_info()
    if args.device:
        device = args.device

    data_yaml = resolve_data_yaml(args)
    train(args, data_yaml, device)


if __name__ == "__main__":
    main()
