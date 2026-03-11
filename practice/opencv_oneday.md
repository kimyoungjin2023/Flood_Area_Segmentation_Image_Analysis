# OpenCV 하루 완성 튜토리얼
> CV 개발자 입문 - 핵심만 빠르게

```bash
pip install opencv-python numpy matplotlib
```

---

## 📌 공통 유틸 - 이미지 출력 함수

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(title, img, cmap=None):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## PART 1. 이미지 읽기 / 저장 / 색상 변환

```python
# OpenCV는 이미지를 BGR로 읽음 (RGB 아님! 주의)
img_bgr  = cv2.imread("sample.jpg")
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    # → RGB (matplotlib용)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # → 흑백

# 저장
cv2.imwrite("output_gray.jpg", img_gray)

# shape 확인
print(f"BGR shape : {img_bgr.shape}")   # (H, W, 3)
print(f"Gray shape: {img_gray.shape}")  # (H, W)

show("원본 RGB", img_rgb)
show("흑백", img_gray, cmap='gray')
```

> ⭐ **핵심 포인트**
> - `cv2.imread` → **BGR** 순서로 읽음
> - `plt.imshow` → **RGB** 기대
> - 항상 `cvtColor(BGR2RGB)` 후 show 해야 색이 정상으로 나옴
> - `img_rgb = img_bgr[:, :, ::-1]` 이렇게 채널 뒤집기도 가능

---

## PART 2. 이진화 (Thresholding)

```python
# 단순 이진화 - 픽셀값 127 기준으로 0 or 255
_, thresh_simple = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Otsu 이진화 - 최적 임계값 자동 계산 (추천!)
_, thresh_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive 이진화 - 영역별로 임계값 다르게 (조명 불균일할 때 좋음)
thresh_adapt = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

show("단순 이진화", thresh_simple, cmap='gray')
show("Otsu 이진화", thresh_otsu, cmap='gray')
show("Adaptive 이진화", thresh_adapt, cmap='gray')
```

> ⭐ **핵심 포인트**
> - **Otsu**: 임계값 자동 계산 → 홍수 영역(물) 분리할 때 자주 씀
> - **Adaptive**: CCTV처럼 조명 변화 심한 영상에 적합

---

## PART 3. 엣지 검출 (Edge Detection)

```python
# Canny - 가장 많이 쓰는 엣지 검출기
# 두 숫자: 낮은 임계값, 높은 임계값 (보통 1:2 또는 1:3 비율)
edges = cv2.Canny(img_gray, 50, 150)

# Sobel - X/Y 방향 그라디언트 따로 계산
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # X방향
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # Y방향
sobel_mag = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

show("Canny 엣지", edges, cmap='gray')
show("Sobel 크기", sobel_mag, cmap='gray')
```

> ⭐ **핵심 포인트**
> - **Canny**: 노이즈에 강하고 얇은 엣지 → 윤곽선 추출에 최적
> - **Sobel**: 방향 정보 포함 → 그라디언트 분석할 때 사용

---

## PART 4. 마스킹 (Masking)

```python
# 마스크 = 관심 영역만 0/255로 표현한 흑백 이미지
# 홍수 segmentation에서 "물 영역 = 255, 나머지 = 0" 이 마스크임

# 예시: 이미지 중앙 원형 영역만 보기
mask = np.zeros(img_gray.shape, dtype=np.uint8)
h, w = img_gray.shape
cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)

# 마스크 적용 (AND 연산)
masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
show("마스크 적용 결과", masked_img)

# HSV 색상 기반 마스킹 (특정 색 추출)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 파란색 계열 추출 (물, 하늘 등)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue  = cv2.inRange(img_hsv, lower_blue, upper_blue)
masked_blue = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_blue)
show("파란색 마스킹", masked_blue)
```

> ⭐ **핵심 포인트**
> - HSV 마스킹은 홍수 탐지 **rule-based 초기 접근법**으로 많이 씀
> - 딥러닝 전에 물 영역 대충 뽑을 때 유용
> - **H**: 색상(Hue) | **S**: 채도(Saturation) | **V**: 밝기(Value)

---

## PART 5. 컨투어 추출 (Contour)

```python
# 컨투어 = 마스크에서 물체의 경계선 좌표 추출
contours, hierarchy = cv2.findContours(
    thresh_otsu,
    cv2.RETR_EXTERNAL,       # 가장 바깥 컨투어만
    cv2.CHAIN_APPROX_SIMPLE  # 꼭짓점만 저장 (메모리 절약)
)

# 컨투어 그리기
contour_img = img_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
show("컨투어 추출", contour_img)

# 면적 기준으로 큰 객체만 필터링 (노이즈 제거)
min_area = 500
large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
print(f"전체 컨투어: {len(contours)}개, 필터 후: {len(large_contours)}개")

# 각 컨투어의 바운딩 박스 그리기
bbox_img = img_rgb.copy()
for c in large_contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
show("바운딩 박스", bbox_img)
```

> ⭐ **핵심 포인트**
> - Segmentation 마스크 → 컨투어 추출 → 바운딩박스 변환
> - 이 흐름이 **Seg 결과를 Detection처럼 활용**하는 방법

---

## PART 6. 동영상 / CCTV 프레임 처리

```python
def process_video(video_path: str, output_path: str):
    """
    영상을 읽어서 프레임마다 처리 후 저장하는 파이프라인
    CCTV 분석의 기본 구조
    """
    cap = cv2.VideoCapture(video_path)  # 0이면 웹캠

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, 해상도: {width}x{height}, 총 프레임: {total}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()   # ret: 읽기 성공 여부, frame: BGR
        if not ret:
            break

        # ── 여기서 프레임별 처리 ──────────────────────
        gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges       = cv2.Canny(gray, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result      = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)
        # ─────────────────────────────────────────────

        cv2.putText(result, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(result)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"처리 중: {frame_idx}/{total}")

    cap.release()
    out.release()
    print(f"완료! 저장: {output_path}")

# process_video("input.mp4", "output.mp4")
```

> ⭐ **핵심 포인트**
> - `VideoCapture(0)` → 웹캠 실시간
> - `VideoCapture("파일경로")` → 영상 파일
> - while 루프 안 `# 여기서 처리` 부분에 **딥러닝 모델 추론 코드** 넣으면 실시간 분석 완성

---

## PART 7. 노이즈 제거 & 모폴로지 연산

```python
# 블러 (노이즈 제거)
blur_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)   # 자연스러운 블러
blur_median   = cv2.medianBlur(img_gray, 5)               # 소금후추 노이즈에 강함

# 모폴로지 - 마스크 정제할 때 필수
kernel = np.ones((5, 5), np.uint8)

dilated = cv2.dilate(thresh_otsu, kernel, iterations=1)                        # 팽창 - 마스크 키우기
eroded  = cv2.erode(thresh_otsu, kernel, iterations=1)                         # 침식 - 마스크 줄이기
opened  = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)                # 침식→팽창 (작은 노이즈 제거)
closed  = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)               # 팽창→침식 (구멍 메우기)

show("팽창", dilated, cmap='gray')
show("침식", eroded, cmap='gray')
show("열기 - 노이즈 제거", opened, cmap='gray')
show("닫기 - 구멍 메우기", closed, cmap='gray')
```

| 연산 | 효과 | 홍수 분석 활용 |
|------|------|---------------|
| `MORPH_OPEN` | 작은 노이즈 제거 | 모델 출력 잡티 제거 |
| `MORPH_CLOSE` | 구멍 메우기 | 물 영역 끊김 복원 |
| `dilate` | 마스크 팽창 | 경계 확장 |
| `erode` | 마스크 침식 | 경계 축소 |

---

## PART 8. 실전 미니 파이프라인 - 홍수 마스크 후처리

> 딥러닝 모델 출력 마스크를 정제하는 함수 - **내일 U-Net 붙일 때 바로 연결됩니다**

```python
def postprocess_flood_mask(pred_mask: np.ndarray) -> np.ndarray:
    """
    딥러닝 모델이 출력한 홍수 segmentation 마스크를 정제하는 함수

    Args:
        pred_mask: 모델 예측 마스크 (0~255 또는 0~1)
    Returns:
        정제된 바이너리 마스크
    """
    # 0~1이면 0~255로 변환
    if pred_mask.max() <= 1.0:
        pred_mask = (pred_mask * 255).astype(np.uint8)

    # 1. 이진화
    _, binary = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)

    # 2. 노이즈 제거 (작은 점 제거)
    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. 구멍 메우기
    filled  = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. 작은 컨투어 제거 (면적 기준)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(filled)
    for c in contours:
        if cv2.contourArea(c) > 1000:   # 1000px² 이상만 유지
            cv2.drawContours(result, [c], -1, 255, -1)

    return result

# 사용 예시
# raw_mask   = model_output          # 딥러닝 모델 출력
# clean_mask = postprocess_flood_mask(raw_mask)
```

---

## ✅ 오늘 배운 것 총정리

| # | 주제 | 핵심 함수 |
|---|------|-----------|
| 1 | 이미지 읽기/변환 | `imread`, `cvtColor`, `imwrite` |
| 2 | 이진화 | `threshold`, `adaptiveThreshold` |
| 3 | 엣지 검출 | `Canny`, `Sobel` |
| 4 | 마스킹 | `bitwise_and`, `inRange` |
| 5 | 컨투어 | `findContours`, `drawContours`, `boundingRect` |
| 6 | 동영상 처리 | `VideoCapture`, `VideoWriter` |
| 7 | 모폴로지 | `dilate`, `erode`, `morphologyEx` |
| 8 | 실전 파이프라인 | `postprocess_flood_mask` |

---

## 🗓️ 내일 할 것 (Week 2 시작)

```
홍수 데이터셋 → U-Net 학습 → 예측 마스크 출력
                                    ↓
                         postprocess_flood_mask 적용
                                    ↓
                            mIoU 성능 측정
```
