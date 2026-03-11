"""
data/transforms.py
==================
학습/검증용 데이터 증강 정의
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.config import cfg


def get_transforms(phase: str) -> A.Compose:
    """
    학습/검증 단계별 transform 반환

    Args:
        phase: "train" 또는 "val"
    Returns:
        Albumentations Compose 객체
    """
    if phase == "train":
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            # 기하학적 변환
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2,
                rotate_limit=30, p=0.5
            ),
            # 색상 변환
            A.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1, p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            # 정규화 + Tensor
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    else:  # val / test
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
