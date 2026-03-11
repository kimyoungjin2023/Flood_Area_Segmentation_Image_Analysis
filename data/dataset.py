"""
data/dataset.py
===============
FloodDataset 클래스 및 DataLoader 생성 함수
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config.config import cfg
from data.transforms import get_transforms


def read_image(path: str) -> np.ndarray:
    """
    Windows 경로 문제 없이 이미지 읽기 (BGR)
    """
    return cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )


def read_mask(path: str) -> np.ndarray:
    """
    마스크를 흑백으로 읽기
    """
    return cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )


def binarize_mask(mask: np.ndarray, threshold: int = None) -> np.ndarray:
    """
    마스크 이진화
    경계 픽셀(중간값)을 0 또는 1로 명확하게 변환

    Args:
        mask     : 원본 마스크 (0~255)
        threshold: 이진화 기준값 (기본: cfg.MASK_THRESHOLD)
    Returns:
        float32 배열 (0.0 또는 1.0)
    """
    threshold = threshold or cfg.MASK_THRESHOLD
    return (mask >= threshold).astype(np.float32)


class FloodDataset(Dataset):
    """
    홍수 Segmentation 데이터셋

    폴더 구조:
        dataset/image/  → 원본 이미지 (.jpg)
        dataset/Mask/   → 마스크 이미지 (.png)
    """

    def __init__(self, img_paths: list, mask_paths: list, transform=None):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # 이미지 읽기 (BGR → RGB)
        img = read_image(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 마스크 읽기 + 이진화
        mask = read_mask(self.mask_paths[idx])
        mask = binarize_mask(mask)

        # Augmentation 적용
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]
            mask = augmented["mask"]

        # 마스크 차원 추가 (H, W) → (1, H, W)
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)

        return img, mask


def get_file_paths(image_dir: str, mask_dir: str):
    """
    이미지/마스크 파일 경로 리스트 반환

    Returns:
        (img_paths, mask_paths) 튜플
    """
    img_files  = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    img_paths  = [os.path.join(image_dir, f) for f in img_files]
    mask_paths = [os.path.join(mask_dir, f)  for f in mask_files]

    return img_paths, mask_paths


def split_dataset(img_paths: list, mask_paths: list, val_ratio: float = None):
    """
    Train / Val 분할

    Args:
        img_paths  : 이미지 경로 리스트
        mask_paths : 마스크 경로 리스트
        val_ratio  : 검증 비율 (기본: cfg.VAL_RATIO)
    Returns:
        (train_img, train_mask, val_img, val_mask)
    """
    val_ratio = val_ratio or cfg.VAL_RATIO
    n = len(img_paths)
    n_val = int(n * val_ratio)

    indices   = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx   = indices[:n_val]

    train_img  = [img_paths[i]  for i in train_idx]
    train_mask = [mask_paths[i] for i in train_idx]
    val_img    = [img_paths[i]  for i in val_idx]
    val_mask   = [mask_paths[i] for i in val_idx]

    print(f"Train: {len(train_img)}장 | Val: {len(val_img)}장")
    return train_img, train_mask, val_img, val_mask


def build_dataloaders(image_dir: str = None, mask_dir: str = None):
    """
    Train / Val DataLoader 생성

    Args:
        image_dir: 이미지 폴더 경로 (기본: cfg.IMAGE_DIR)
        mask_dir : 마스크 폴더 경로 (기본: cfg.MASK_DIR)
    Returns:
        (train_loader, val_loader)
    """
    image_dir = image_dir or cfg.IMAGE_DIR
    mask_dir  = mask_dir  or cfg.MASK_DIR

    img_paths, mask_paths = get_file_paths(image_dir, mask_dir)
    train_img, train_mask, val_img, val_mask = split_dataset(
        img_paths, mask_paths
    )

    train_ds = FloodDataset(train_img, train_mask, get_transforms("train"))
    val_ds   = FloodDataset(val_img,   val_mask,   get_transforms("val"))

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )

    return train_loader, val_loader
