"""
config/config.py
================
프로젝트 전체 설정을 한 곳에서 관리
"""

import torch


class Config:
    # ── 경로 ──────────────────────────────────────
    IMAGE_DIR  = "./dataset/image"
    MASK_DIR   = "./dataset/Mask"
    SAVE_DIR   = "./checkpoints"
    LOG_DIR    = "./logs"

    # ── 이미지 ─────────────────────────────────────
    IMG_SIZE   = 512
    MASK_THRESHOLD = 128      # 마스크 이진화 기준값

    # ── 학습 ──────────────────────────────────────
    BATCH_SIZE = 8            # GPU 메모리 부족하면 4로 낮추기
    EPOCHS     = 30
    LR         = 1e-4
    VAL_RATIO  = 0.2
    SEED       = 42

    # ── 모델 ──────────────────────────────────────
    ENCODER        = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES    = 1         # 이진 분류

    # ── 디바이스 ───────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 시각화 ─────────────────────────────────────
    FONT_FAMILY = "Malgun Gothic"   # Windows 한글 폰트


cfg = Config()
