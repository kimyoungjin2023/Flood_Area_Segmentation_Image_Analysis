"""
단순 읽기 테스트 - 이것만 먼저 실행해보세요
"""
import os
import cv2
import numpy as np

IMAGE_DIR = "./dataset/image"
MASK_DIR  = "./dataset/Mask"

# 첫 번째 파일 경로 직접 출력
img_files  = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

img_path  = os.path.join(IMAGE_DIR, img_files[0])
mask_path = os.path.join(MASK_DIR, mask_files[0])

print(f"이미지 경로: {img_path}")
print(f"마스크 경로: {mask_path}")
print(f"이미지 파일 존재: {os.path.exists(img_path)}")
print(f"마스크 파일 존재: {os.path.exists(mask_path)}")
print(f"이미지 파일 크기: {os.path.getsize(img_path)} bytes")
print()

# 방법 1: cv2.imread
img1 = cv2.imread(img_path)
print(f"방법1 cv2.imread         : {img1}")

# 방법 2: imdecode
img2 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"방법2 imdecode+fromfile  : {type(img2)}, shape={img2.shape if img2 is not None else 'None'}")

# 방법 3: PIL
try:
    from PIL import Image
    img3 = np.array(Image.open(img_path))
    print(f"방법3 PIL                : shape={img3.shape}")
except ImportError:
    print("방법3 PIL: 설치 필요 → pip install Pillow")
