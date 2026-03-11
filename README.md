# 🌊 Flood Area Segmentation

홍수 영역 탐지를 위한 U-Net 기반 Semantic Segmentation 프로젝트

---

## 📁 프로젝트 구조

```
flood_segmentation/
│
├── main.py                  ← 진입점 (학습/평가/예측 실행)
│
├── config/
│   └── config.py            ← 전체 설정 (경로, 하이퍼파라미터)
│
├── data/
│   ├── dataset.py           ← FloodDataset 클래스, DataLoader 생성
│   └── transforms.py        ← 학습/검증 데이터 증강
│
├── models/
│   └── unet.py              ← U-Net 모델 빌드 및 로드
│
├── utils/
│   ├── losses.py            ← DiceBCELoss, FocalLoss 등
│   ├── metrics.py           ← IoU, Dice, PixelAcc 계산
│   └── visualize.py         ← 학습 곡선, 예측 결과 시각화
│
├── train/
│   └── trainer.py           ← 학습/검증 루프, 체크포인트 저장
│
└── evaluate/
    └── evaluator.py         ← 모델 평가, 예측 결과 시각화
```

---

## 🗂️ 데이터셋 구조

```
dataset/
  image/    ← 원본 홍수 이미지 (.jpg)  290장
  Mask/     ← 이진 마스크 (.png)       290장
             (0 = 배경, 255 = 홍수 영역)
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 실행

### 학습
```bash
python main.py
# main.py 내 mode="train" (기본값)
```

### 평가
```python
# main.py 하단 수정
main(mode="evaluate")
```

### 단일 이미지 예측
```python
main(mode="predict")
# evaluator.py의 image_path를 원하는 이미지로 변경
```

---

## 📊 성능 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| IoU | 예측/정답 마스크 겹침 비율 | 0.7 이상 |
| Dice Score | F1 기반 영역 유사도 | 0.8 이상 |
| Pixel Acc | 픽셀 단위 정확도 | 0.9 이상 |

---

## 🧪 모델 설정

| 항목 | 값 |
|------|-----|
| 모델 | U-Net |
| 백본 | ResNet-34 (ImageNet pretrained) |
| 입력 크기 | 512 × 512 |
| 손실 함수 | DiceBCELoss |
| 옵티마이저 | Adam (lr=1e-4) |
| 스케줄러 | ReduceLROnPlateau |

---

## 🔧 설정 변경

`config/config.py` 에서 모든 설정 변경 가능

```python
cfg.BATCH_SIZE = 4        # GPU 메모리 부족 시
cfg.EPOCHS     = 50       # 더 오래 학습
cfg.ENCODER    = "resnet50"  # 더 강력한 백본
```
