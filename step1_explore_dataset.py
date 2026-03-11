"""
홍수 데이터셋 탐색 코드 (Step 1)
==============================
이 코드를 먼저 실행해서 데이터 구조를 파악하세요.

폴더 구조 가정:
  dataset/
    images/   ← 원본 이미지 (.jpg 또는 .png)
    masks/    ← 마스크 이미지 (.png, 흑백)

설치:
  pip install opencv-python numpy matplotlib
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# ============================================================
# ⚙️ 설정 - 본인 데이터셋 경로로 바꿔주세요
# ============================================================
IMAGE_DIR = "./dataset/image"
MASK_DIR  = "./dataset/Mask"


# ============================================================
# 🔧 Windows 경로 대응 imread 래퍼
# ============================================================
def imread_safe(path, flag=cv2.IMREAD_COLOR):
    """cv2.imread 대신 사용 - Windows 경로 문제 안전하게 처리"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)


# ============================================================
# STEP 1. 파일 목록 확인
# ============================================================
def step1_check_files():
    print("=" * 50)
    print("STEP 1. 파일 목록 확인")
    print("=" * 50)

    img_files  = sorted(os.listdir(IMAGE_DIR))
    mask_files = sorted(os.listdir(MASK_DIR))

    print(f"이미지 수 : {len(img_files)}장")
    print(f"마스크 수 : {len(mask_files)}장")
    print(f"\n이미지 샘플 (앞 5개): {img_files[:5]}")
    print(f"마스크 샘플 (앞 5개): {mask_files[:5]}")

    # 이미지-마스크 쌍이 맞는지 확인
    img_stems  = set(os.path.splitext(f)[0] for f in img_files)
    mask_stems = set(os.path.splitext(f)[0] for f in mask_files)

    only_img  = img_stems - mask_stems
    only_mask = mask_stems - img_stems

    if only_img:
        print(f"\n⚠️  마스크 없는 이미지: {len(only_img)}개 → {list(only_img)[:5]}")
    if only_mask:
        print(f"⚠️  이미지 없는 마스크: {len(only_mask)}개 → {list(only_mask)[:5]}")
    if not only_img and not only_mask:
        print("\n✅ 이미지-마스크 쌍 완벽히 일치!")

    return img_files, mask_files


# ============================================================
# STEP 2. 이미지/마스크 기본 정보 확인
# ============================================================
def step2_check_shape(img_files, mask_files):
    print("\n" + "=" * 50)
    print("STEP 2. 이미지/마스크 크기 및 타입 확인")
    print("=" * 50)

    # 첫 번째 샘플로 확인
    img_path  = os.path.join(IMAGE_DIR, img_files[0])
    mask_path = os.path.join(MASK_DIR, mask_files[0])

    img  = imread_safe(img_path)
    mask = imread_safe(mask_path, cv2.IMREAD_GRAYSCALE)

    print(f"이미지 shape : {img.shape}  (H, W, C)")
    print(f"마스크 shape : {mask.shape}  (H, W)")
    print(f"이미지 dtype : {img.dtype}")
    print(f"마스크 dtype : {mask.dtype}")
    print(f"마스크 최솟값: {mask.min()}")
    print(f"마스크 최댓값: {mask.max()}")

    # 크기가 다양한지 확인 (여러 장 샘플링)
    shapes = []
    for f in img_files[:20]:
        i = imread_safe(os.path.join(IMAGE_DIR, f))
        if i is not None:
            shapes.append(i.shape)
    unique_shapes = set(shapes)
    print(f"\n이미지 해상도 종류 (샘플 20장): {unique_shapes}")
    if len(unique_shapes) > 1:
        print("⚠️  해상도가 다양함 → 학습 전 resize 통일 필요!")
    else:
        print("✅ 모든 이미지 해상도 동일!")

    return img, mask


# ============================================================
# STEP 3. 클래스(픽셀값) 분석 ← 핵심!
# ============================================================
def step3_check_classes(mask_files):
    print("\n" + "=" * 50)
    print("STEP 3. 마스크 픽셀값 분석 (클래스 확인)")
    print("=" * 50)

    all_values = Counter()

    # 전체 마스크 픽셀값 수집
    for f in mask_files:
        mask = imread_safe(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique_vals = np.unique(mask)
        for v in unique_vals:
            all_values[int(v)] += 1

    print("마스크에 존재하는 픽셀값 (클래스 후보):")
    print(f"{'픽셀값':>8} | {'등장한 이미지 수':>15} | 의미 추정")
    print("-" * 50)

    for val, count in sorted(all_values.items()):
        # 픽셀값으로 클래스 추정
        if val == 0:
            guess = "배경 (Background)"
        elif val == 255:
            guess = "관심 클래스 (홍수/물 가능성 높음)"
        elif val == 1:
            guess = "클래스 1 (정수 인코딩 방식)"
        elif val == 2:
            guess = "클래스 2 (정수 인코딩 방식)"
        else:
            guess = f"클래스 {val}"
        print(f"{val:>8} | {count:>15}장 | {guess}")

    print("\n💡 해석 방법:")
    print("  픽셀값 0, 255만 있음    → 이진 마스크 (홍수 vs 배경)")
    print("  픽셀값 0, 1, 2... 있음  → 정수 인코딩 (클래스별 번호)")
    print("  픽셀값이 다양          → 팔레트 이미지일 수 있음")

    return all_values


# ============================================================
# STEP 4. 샘플 시각화
# ============================================================
def step4_visualize_samples(img_files, mask_files, n_samples=4):
    print("\n" + "=" * 50)
    print("STEP 4. 샘플 시각화")
    print("=" * 50)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    if n_samples == 1:
        axes = [axes]

    for i in range(min(n_samples, len(img_files))):
        img  = imread_safe(os.path.join(IMAGE_DIR, img_files[i]))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = imread_safe(os.path.join(MASK_DIR, mask_files[i]), cv2.IMREAD_GRAYSCALE)

        # 마스크 오버레이 (원본 위에 반투명하게)
        overlay = img.copy()
        # 마스크가 255인 곳을 빨간색으로 표시
        mask_region = mask > 0
        overlay[mask_region] = (
            overlay[mask_region] * 0.5 + np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)

        axes[i][0].imshow(img)
        axes[i][0].set_title(f"원본 이미지\n{img_files[i]}", fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(mask, cmap='gray')
        axes[i][1].set_title(f"마스크\n고유값: {np.unique(mask)}", fontsize=9)
        axes[i][1].axis('off')

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("오버레이\n(빨간색=홍수 영역)", fontsize=9)
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig("dataset_check.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ 시각화 저장 완료: dataset_check.png")


# ============================================================
# STEP 5. 클래스 불균형 확인
# ============================================================
def step5_check_balance(mask_files):
    print("\n" + "=" * 50)
    print("STEP 5. 클래스 불균형 확인")
    print("=" * 50)

    flood_pixels = 0
    bg_pixels    = 0

    for f in mask_files:
        mask = imread_safe(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        flood_pixels += np.sum(mask > 0)
        bg_pixels    += np.sum(mask == 0)

    total = flood_pixels + bg_pixels
    flood_ratio = flood_pixels / total * 100
    bg_ratio    = bg_pixels    / total * 100

    print(f"홍수 픽셀 비율 : {flood_ratio:.1f}%")
    print(f"배경 픽셀 비율 : {bg_ratio:.1f}%")

    # 불균형 경고
    if flood_ratio < 10:
        print("\n⚠️  심각한 클래스 불균형!")
        print("   → 학습 시 손실 함수에 가중치 필요 (pos_weight 설정)")
        print("   → Dice Loss 또는 Focal Loss 사용 권장")
    elif flood_ratio < 30:
        print("\n⚠️  클래스 불균형 있음")
        print("   → Dice Loss 사용 권장")
    else:
        print("\n✅ 클래스 균형 양호!")

    # 파이 차트
    plt.figure(figsize=(6, 6))
    plt.pie(
        [flood_pixels, bg_pixels],
        labels=[f'홍수({flood_ratio:.1f}%)', f'배경({bg_ratio:.1f}%)'],
        colors=['#3498db', '#95a5a6'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("클래스 픽셀 분포")
    plt.savefig("class_balance.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ 분포 차트 저장 완료: class_balance.png")


# ============================================================
# STEP 6. 다음 단계 안내
# ============================================================
def step6_next_guide(all_values):
    print("\n" + "=" * 50)
    print("STEP 6. 탐색 완료 → 다음 단계 안내")
    print("=" * 50)

    vals = sorted(all_values.keys())

    if vals == [0, 255]:
        print("📌 마스크 타입: 이진 마스크 (Binary)")
        print("   → num_classes = 1")
        print("   → 손실함수: BCEWithLogitsLoss 또는 DiceLoss")
        print("   → 모델 출력: sigmoid 활성화")

    elif all(v < 20 for v in vals):
        n = max(vals) + 1
        print(f"📌 마스크 타입: 정수 인코딩 (Multi-class, {n}개 클래스)")
        print(f"   → num_classes = {n}")
        print("   → 손실함수: CrossEntropyLoss")
        print("   → 모델 출력: softmax 활성화")

    else:
        print("📌 마스크 타입: 불명확 → 직접 확인 필요")
        print("   → STEP 4 시각화 결과 보고 판단하세요")

    print("\n✅ 탐색 완료! 다음 단계:")
    print("   Step 2 → Dataset 클래스 + DataLoader 작성")
    print("   Step 3 → U-Net 모델 학습")
    print("   Step 4 → 결과 시각화 + mIoU 측정")


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    img_files, mask_files = step1_check_files()
    img, mask             = step2_check_shape(img_files, mask_files)
    all_values            = step3_check_classes(mask_files)
    step4_visualize_samples(img_files, mask_files, n_samples=4)
    step5_check_balance(mask_files)
    step6_next_guide(all_values)
