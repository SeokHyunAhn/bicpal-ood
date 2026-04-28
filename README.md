# Orchard OOD Detection

자율주행 농기계의 **OOD(Out-of-Distribution) 장애물 탐지** 연구.

- **기반 모델**: YOLOv8 — 사람·손수레·사다리 3클래스 탐지 (목표 mAP ≥ 95%)
- **OOD 탐지**: Energy Score — 운반차·트럭·방제기 탐지 (목표 AUROC ≥ 0.90)
- **데이터**: NIA AI Hub 과수원 내 로봇 주행 데이터 (사과·배, 505,581장)

## 파이프라인

```
1. 데이터 필터링     data/filter_ood_images.py
2. 데이터셋 구성     data/build_yolo_dataset.py
3. YOLOv8 학습       yolov8/train.py
4. Energy Score 평가 (예정)
```

## 빠른 시작

```bash
pip install -r requirements.txt

# 1. OOD 오염 이미지 목록 생성
python data/filter_ood_images.py

# 2. YOLO 포맷 데이터셋 구성 (하드링크, 복사 없음)
python data/build_yolo_dataset.py

# 3. 학습 (데이터셋 경로 지정)
python yolov8/train.py --dataset-root D:/orchard_data/yolo_dataset
```

### 다른 컴퓨터에서 실행

```bash
python yolov8/train.py --dataset-root <yolo_dataset 경로>
```

## 데이터셋 클래스

| YOLO 클래스 | 원본 category_id | 이름 | 역할 |
|-------------|-----------------|------|------|
| 0 | 1 | 사람 | In-Distribution |
| 1 | 2 | 손수레 | In-Distribution |
| 2 | 6 | 사다리 | In-Distribution |
| — | 3 | 운반차 | OOD (학습 제외) |
| — | 4 | 트럭 | OOD (학습 제외) |
| — | 5 | 방제기 | OOD (학습 제외) |

## 필터링 결과

| 구분 | 전체 | OOD 제외 | 학습 사용 |
|------|------|---------|---------|
| Training | 311,016 | 34,872 | 276,144 |
| Validation | 41,969 | 13,732 | 28,237 |
