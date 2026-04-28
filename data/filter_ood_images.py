"""
OOD 오염 방지 필터링 스크립트

BBX JSON을 읽어 category_id {3, 4, 5}가 포함된 이미지를 학습셋에서 제거할
파일 목록을 생성한다.

출력:
  - ood_excluded_images.txt : 제외할 이미지 경로 목록 (절대 경로)
  - filter_stats.json       : 통계 요약
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_ROOT = Path(r"D:\orchard_data")
SPLITS = {
    "Training": (DATA_ROOT / "Training" / "TL", DATA_ROOT / "Training" / "TS"),
    "Validation": (DATA_ROOT / "Validation" / "VL", DATA_ROOT / "Validation" / "VS"),
}
FRUIT_TYPES = ["Apple", "Pear"]
OOD_CATEGORY_IDS = {3, 4, 5}  # 운반차, 트럭, 방제기

OUTPUT_DIR = Path(__file__).parent
EXCLUDED_LIST_PATH = OUTPUT_DIR / "ood_excluded_images.txt"
STATS_PATH = OUTPUT_DIR / "filter_stats.json"


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def load_json_utf8(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bbx_json_to_img_path(bbx_json: Path, ts_root: Path, fruit: str) -> Path | None:
    """BBX JSON 경로 → 대응 이미지(IMG) 경로로 변환."""
    # bbx_json: TL/Apple/{session}/BBX/{name}_BBX.json
    session = bbx_json.parts[-3]          # 세션명
    stem = bbx_json.stem.replace("_BBX", "")
    img_path = ts_root / fruit / session / "IMG" / f"{stem}_IMG.jpg"
    return img_path


def find_bbx_jsons(tl_root: Path, fruit: str) -> list[Path]:
    fruit_dir = tl_root / fruit
    if not fruit_dir.exists():
        return []
    return sorted(fruit_dir.rglob("*_BBX.json"))


# ──────────────────────────────────────────────
# 메인 처리
# ──────────────────────────────────────────────

def scan_split(split_name: str, tl_root: Path, ts_root: Path) -> tuple[list[str], dict]:
    """한 split(Training / Validation)을 스캔, 제외할 이미지 절대경로 목록 반환."""
    excluded_paths: list[str] = []
    stats = {
        "total_jsons": 0,
        "ood_jsons": 0,
        "ood_by_category": defaultdict(int),
        "missing_image": 0,
    }

    for fruit in FRUIT_TYPES:
        jsons = find_bbx_jsons(tl_root, fruit)
        stats["total_jsons"] += len(jsons)

        for bbx_path in jsons:
            try:
                data = load_json_utf8(bbx_path)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[WARN] JSON 읽기 실패: {bbx_path} — {e}")
                continue

            annotations = data.get("annotations", [])
            ood_cats_in_file = {
                ann["category_id"]
                for ann in annotations
                if ann.get("category_id") in OOD_CATEGORY_IDS
            }

            if not ood_cats_in_file:
                continue

            # OOD 오염 이미지
            stats["ood_jsons"] += 1
            for cat_id in ood_cats_in_file:
                stats["ood_by_category"][cat_id] += 1

            img_path = bbx_json_to_img_path(bbx_path, ts_root, fruit)
            if not img_path.exists():
                stats["missing_image"] += 1
                # 이미지가 없어도 경로는 기록 (나중에 확인용)
                print(f"[WARN] 이미지 없음: {img_path}")

            excluded_paths.append(str(img_path))

    return excluded_paths, stats


def main():
    all_excluded: list[str] = []
    all_stats: dict = {}

    for split_name, (tl_root, ts_root) in SPLITS.items():
        print(f"\n── {split_name} 스캔 중... (TL: {tl_root})")
        if not tl_root.exists():
            print(f"   [SKIP] 경로 없음: {tl_root}")
            continue

        excluded, stats = scan_split(split_name, tl_root, ts_root)
        all_excluded.extend(excluded)

        cat_names = {3: "운반차", 4: "트럭", 5: "방제기"}
        print(f"   전체 BBX JSON : {stats['total_jsons']:,}")
        print(f"   OOD 포함 JSON : {stats['ood_jsons']:,}")
        for cat_id, count in sorted(stats["ood_by_category"].items()):
            print(f"     cat {cat_id} ({cat_names[cat_id]}): {count:,}개 파일")
        if stats["missing_image"]:
            print(f"   이미지 미존재 : {stats['missing_image']:,}")

        all_stats[split_name] = {
            **stats,
            "ood_by_category": dict(stats["ood_by_category"]),
        }

    # 중복 제거 및 정렬
    unique_excluded = sorted(set(all_excluded))

    # 출력
    EXCLUDED_LIST_PATH.write_text(
        "\n".join(unique_excluded), encoding="utf-8"
    )
    STATS_PATH.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✓ 제외 이미지 목록: {EXCLUDED_LIST_PATH}  ({len(unique_excluded):,}개)")
    print(f"✓ 통계 파일:        {STATS_PATH}")


if __name__ == "__main__":
    main()
