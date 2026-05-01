"""
OOD 테스트셋 준비 스크립트

ood_excluded_images.txt에서 Validation 분할의 이미지만 추려서
data/ood_test_images.txt로 저장한다.
"""

from pathlib import Path

EXCLUDED_PATH = Path(__file__).parent.parent / "data" / "ood_excluded_images.txt"
OUTPUT_PATH   = Path(__file__).parent.parent / "data" / "ood_test_images.txt"


def main():
    lines     = EXCLUDED_PATH.read_text(encoding="utf-8").splitlines()
    all_paths = [l.strip() for l in lines if l.strip()]

    val_ood  = [p for p in all_paths if "Validation" in p]
    existing = [p for p in val_ood if Path(p).exists()]
    missing  = len(val_ood) - len(existing)

    OUTPUT_PATH.write_text("\n".join(existing), encoding="utf-8")

    print(f"전체 OOD 이미지  : {len(all_paths):,}")
    print(f"Validation OOD  : {len(val_ood):,}")
    print(f"실제 존재        : {len(existing):,}")
    if missing:
        print(f"[WARN] 파일 없음 : {missing:,}개")
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
