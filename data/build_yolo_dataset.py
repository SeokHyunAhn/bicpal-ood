"""
YOLOv8 학습용 데이터셋 빌드 스크립트

- OOD 오염 이미지(ood_excluded_images.txt)를 제외
- BBX JSON → YOLO .txt 라벨 변환 (category_id 1, 2, 6만 사용)
- 이미지는 하드링크로 연결 (같은 D: 드라이브, 복사 없음)
- dataset.yaml 생성

출력 구조:
  D:/orchard_data/yolo_dataset/
  ├── images/train/   (하드링크)
  ├── images/val/     (하드링크)
  ├── labels/train/   (YOLO .txt)
  ├── labels/val/     (YOLO .txt)
  └── dataset.yaml
"""

import json
import os
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────
DATA_ROOT   = Path(r"D:\orchard_data")
OUTPUT_ROOT = Path(r"D:\orchard_data\yolo_dataset")
SCRIPT_DIR  = Path(__file__).parent

SPLITS = {
    "train": (DATA_ROOT / "Training" / "TL",   DATA_ROOT / "Training" / "TS"),
    "val":   (DATA_ROOT / "Validation" / "VL",  DATA_ROOT / "Validation" / "VS"),
}
FRUIT_TYPES = ["Apple", "Pear"]

# category_id → YOLO 클래스 인덱스 (OOD·미사용 클래스는 키에 없음)
ID_CATEGORY_MAP: dict[int, int] = {1: 0, 2: 1, 6: 2}
CLASS_NAMES = ["사람", "손수레", "사다리"]

OOD_EXCLUDED_PATH = SCRIPT_DIR / "ood_excluded_images.txt"

DATASET_YAML_PATH = OUTPUT_ROOT / "dataset.yaml"


# ─────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────

def load_excluded_set(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"제외 목록 없음: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def load_json_utf8(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO [x_min, y_min, w, h] → YOLO [cx_norm, cy_norm, w_norm, h_norm]"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    # 0~1 클리핑 (간혹 bbox가 이미지 밖으로 벗어나는 노이즈 방지)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return cx, cy, wn, hn


def bbx_path_to_img_path(bbx_path: Path, ts_root: Path, fruit: str) -> Path:
    session = bbx_path.parts[-3]
    stem    = bbx_path.stem.replace("_BBX", "")
    return ts_root / fruit / session / "IMG" / f"{stem}_IMG.jpg"


def progress(current: int, total: int, start: float, prefix: str = ""):
    pct     = current / total * 100
    elapsed = time.time() - start
    rate    = current / elapsed if elapsed > 0 else 0
    eta     = (total - current) / rate if rate > 0 else 0
    bar_len = 30
    filled  = int(bar_len * current / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r{prefix}[{bar}] {pct:5.1f}%  {current:,}/{total:,}"
        f"  {rate:.0f}개/s  ETA {eta/60:.1f}분",
        end="", flush=True,
    )


# ─────────────────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────────────────

def build_split(
    split_name: str,
    tl_root: Path,
    ts_root: Path,
    excluded_set: set[str],
) -> dict:
    img_out_dir = OUTPUT_ROOT / "images" / split_name
    lbl_out_dir = OUTPUT_ROOT / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # 전체 BBX 목록 수집
    all_jsons: list[tuple[str, Path]] = []
    for fruit in FRUIT_TYPES:
        for p in sorted((tl_root / fruit).rglob("*_BBX.json")):
            all_jsons.append((fruit, p))

    total = len(all_jsons)
    stats = {"processed": 0, "skipped_ood": 0, "skipped_no_img": 0,
             "empty_label": 0, "hardlink_exist": 0}

    print(f"\n── {split_name}  ({total:,}개 JSON)")
    start = time.time()

    for i, (fruit, bbx_path) in enumerate(all_jsons, 1):
        if i % 500 == 0 or i == total:
            progress(i, total, start, prefix=f"  {split_name} ")

        img_src = bbx_path_to_img_path(bbx_path, ts_root, fruit)

        # OOD 제외
        if str(img_src) in excluded_set:
            stats["skipped_ood"] += 1
            continue

        # 원본 이미지 존재 확인
        if not img_src.exists():
            stats["skipped_no_img"] += 1
            continue

        # ── 라벨 파일 생성 ──
        lbl_dst = lbl_out_dir / (img_src.stem + ".txt")

        if not lbl_dst.exists():
            try:
                data   = load_json_utf8(bbx_path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # 이미지 크기 (JSON에서 읽기, 기본값 1920×1080)
            images_meta = {im["id"]: im for im in data.get("images", [])}
            anns        = data.get("annotations", [])

            lines: list[str] = []
            for ann in anns:
                cat_id = ann.get("category_id")
                if cat_id not in ID_CATEGORY_MAP:
                    continue
                yolo_cls = ID_CATEGORY_MAP[cat_id]
                img_meta = images_meta.get(ann.get("image_id"), {})
                img_w    = img_meta.get("width", 1920)
                img_h    = img_meta.get("height", 1080)
                cx, cy, wn, hn = coco_to_yolo(ann["bbox"], img_w, img_h)
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

            lbl_dst.write_text("\n".join(lines), encoding="utf-8")
            if not lines:
                stats["empty_label"] += 1
        # ── 이미지 하드링크 ──
        img_dst = img_out_dir / img_src.name
        if not img_dst.exists():
            try:
                os.link(img_src, img_dst)
            except OSError as e:
                # 하드링크 실패 시 복사로 fallback
                import shutil
                shutil.copy2(img_src, img_dst)
        else:
            stats["hardlink_exist"] += 1

        stats["processed"] += 1

    print()  # 개행
    return stats


def write_dataset_yaml():
    content = f"""\
# YOLOv8 dataset configuration
# Generated by build_yolo_dataset.py

path: {OUTPUT_ROOT.as_posix()}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for idx, name in enumerate(CLASS_NAMES):
        content += f"  {idx}: {name}\n"

    DATASET_YAML_PATH.write_text(content, encoding="utf-8")
    print(f"✓ dataset.yaml 생성: {DATASET_YAML_PATH}")


def main():
    print("OOD 제외 목록 로딩 중...")
    excluded_set = load_excluded_set(OOD_EXCLUDED_PATH)
    print(f"  제외 이미지: {len(excluded_set):,}개")

    all_stats: dict[str, dict] = {}
    for split_name, (tl_root, ts_root) in SPLITS.items():
        if not tl_root.exists():
            print(f"[SKIP] 경로 없음: {tl_root}")
            continue
        stats = build_split(split_name, tl_root, ts_root, excluded_set)
        all_stats[split_name] = stats

    print("\n─────────────────── 결과 요약 ───────────────────")
    for split_name, s in all_stats.items():
        print(f"[{split_name}]")
        print(f"  처리 완료   : {s['processed']:,}")
        print(f"  OOD 제외    : {s['skipped_ood']:,}")
        print(f"  이미지 없음 : {s['skipped_no_img']:,}")
        print(f"  빈 라벨     : {s['empty_label']:,}  (ID 클래스 어노테이션 없는 이미지)")

    write_dataset_yaml()
    print(f"\n출력 경로: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
