# 프로젝트 컨텍스트

## 연구 목표

자율주행 농기계의 **OOD(Out-of-Distribution) 장애물 탐지** 알고리즘 개발.

- **기반 모델**: 사람, 손수레, 사다리 3개 클래스 인식
- **OOD 탐지**: 비학습 장애물(운반차, 트럭, 방제기)을 Energy Score로 탐지
- **평가 지표**: 기반 모델 → mAP / OOD 탐지 → AUROC
- **목표 성능**: 기반 모델 mAP ≥ 95%, OOD AUROC ≥ 0.90

## 데이터셋

**NIA AI Hub — 과수원 내 로봇 주행 데이터 (사과, 배 류)**

| 항목 | 내용 |
|------|------|
| 이미지(RGB) | 505,581장 (사과 253,851 / 배 251,730) |
| 포인트클라우드 | 318,832개 — 본 연구 미사용 |
| 라벨링 포맷 | COCO 형식 JSON |
| 학습/테스트/검증 분할 | 80% / 10% / 10% |

### 클래스 ID 매핑

| category_id | 클래스 | 역할 |
|-------------|--------|------|
| 1 | 사람 | ✅ In-Distribution (학습) |
| 2 | 손수레 | ✅ In-Distribution (학습) |
| 6 | 사다리 | ✅ In-Distribution (학습) |
| 3 | 운반차 | ❌ OOD — 학습셋에서 이미지째 제거 |
| 4 | 트럭 | ❌ OOD — 학습셋에서 이미지째 제거 |
| 5 | 방제기 | ❌ OOD — 학습셋에서 이미지째 제거 |
| 7 | 과수박스 | 미사용 |
| 8 | 창고 | 미사용 |
| 9 | 사과 | 미사용 |
| 10 | 배봉지 | 미사용 |
| 11 | 비포장 | 미사용 |
| 12 | 기타 | 미사용 |
| 13 | 사과나무 | 미사용 |
| 14 | 배나무 | 미사용 |

※ JSON 읽기 시 반드시 UTF-8 인코딩 지정할 것 (PowerShell 기본값으로 읽으면 인코딩 깨짐)

### OOD 오염 방지 (핵심 규칙)
OOD 클래스(category_id 3, 4, 5)가 하나라도 포함된 이미지는 **이미지째로** 학습셋에서 제거.
라벨만 지우는 것은 불충분 — 모델이 배경으로 feature를 학습할 수 있음.

## 데이터 디렉토리 구조

```
D:\orchard_data\
├── Training\
│   ├── 01.원천데이터\        # 미사용
│   ├── 02.라벨링데이터\      # 미사용
│   ├── TL\                   # 라벨 (COCO JSON)
│   │   ├── Apple\
│   │   │   └── AP_001_HR_YYYYMMDD_XX_XXX_XXX_XXX\
│   │   │       ├── BBX\      # 바운딩박스 JSON ← 본 연구 사용
│   │   │       └── CUB\      # 3D 큐보이드 JSON (미사용)
│   │   └── Pear\
│   │       └── (동일 구조)
│   └── TS\                   # 원천 이미지
│       ├── Apple\
│       │   └── AP_001_HR_YYYYMMDD_XX_XXX_XXX_XXX\
│       │       ├── IMG\      # JPG 이미지 ← 본 연구 사용
│       │       └── PCD\      # 포인트클라우드 (미사용)
│       └── Pear\
│           └── (동일 구조)
└── Validation\
    ├── VL\                   # 라벨 (TL과 동일 구조)
    └── VS\                   # 원천 이미지 (TS와 동일 구조)
```

### 이미지-라벨 매핑 규칙
- 이미지: `TS/Apple/{세션명}/IMG/{파일명}_IMG.jpg`
- 라벨:   `TL/Apple/{세션명}/BBX/{파일명}_BBX.json`
- 세션명과 파일명 prefix가 동일하므로 이름 기반으로 매핑

### JSON 구조 (COCO 형식)
```json
{
  "categories": [{"id": 1, "name": "..."}],
  "images": [{"id": 1, "width": 1920, "height": 1080,
              "file_name": "AP_001_..._0000_IMG.jpg"}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "bbox": [x, y, width, height],
    "area": 107807.5, "iscrowd": 0
  }]
}
```

## OOD 탐지 방법

**YOLOv8 + Energy Score (Post-hoc)**
재학습 없이 학습된 YOLOv8 logit에서만 계산.

```python
energy = -torch.logsumexp(logits, dim=1)
# 낮으면 In-Distribution, 높으면 OOD
```

## 파이프라인

```
1. 데이터 필터링
   → category_id {3, 4, 5} 포함 이미지를 학습셋에서 이미지째 제거
   → BBX JSON 기준으로 필터링

2. YOLOv8 학습
   → 3클래스: 사람(1), 손수레(2), 사다리(6)
   → 충분한 epoch 학습 (최소 50~100 epoch)

3. Energy Score 적용
   → 임계값 이하: In-Distribution
   → 임계값 이상: OOD (운반차, 트럭, 방제기)

4. 평가
   → mAP (기반 모델) + AUROC (OOD 탐지)
```

## 개발 환경

- **IDE**: VSCode + Claude Code (`--dangerously-skip-permissions` 사용 중)
- **데이터 경로**: `D:\orchard_data\`
- **추출 도구**: 7-Zip (분할 압축 .z01/.z02.../.zip 형식)
- **모델**: YOLOv8
- **이미지 해상도**: 1920×1080
