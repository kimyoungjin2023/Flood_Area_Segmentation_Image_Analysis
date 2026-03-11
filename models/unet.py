"""
models/unet.py
==============
U-Net 모델 빌드 함수
"""

import segmentation_models_pytorch as smp
from config.config import cfg


def build_model(
    encoder_name: str    = None,
    encoder_weights: str = None,
    num_classes: int     = None,
):
    """
    U-Net 모델 생성

    Args:
        encoder_name    : 백본 인코더 (기본: cfg.ENCODER)
        encoder_weights : 사전학습 가중치 (기본: cfg.ENCODER_WEIGHTS)
        num_classes     : 출력 클래스 수 (기본: cfg.NUM_CLASSES)
    Returns:
        U-Net 모델 (DEVICE로 이동 완료)

    사용 가능한 인코더 예시:
        "resnet34"    → 가볍고 빠름 (추천)
        "resnet50"    → 더 강력하지만 느림
        "efficientnet-b3" → 성능/속도 균형 좋음
        "mit_b2"      → Transformer 기반, 높은 성능
    """
    encoder_name    = encoder_name    or cfg.ENCODER
    encoder_weights = encoder_weights or cfg.ENCODER_WEIGHTS
    num_classes     = num_classes     or cfg.NUM_CLASSES

    model = smp.Unet(
        encoder_name    = encoder_name,
        encoder_weights = encoder_weights,
        in_channels     = 3,
        classes         = num_classes,
        activation      = None,    # Loss 함수에서 sigmoid 처리
    )

    print(f"모델 생성 완료: U-Net + {encoder_name} ({encoder_weights})")
    return model.to(cfg.DEVICE)


def load_model(checkpoint_path: str):
    """
    저장된 체크포인트에서 모델 로드

    Args:
        checkpoint_path: .pth 파일 경로
    Returns:
        로드된 모델
    """
    import torch
    model = build_model()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=cfg.DEVICE)
    )
    model.eval()
    print(f"모델 로드 완료: {checkpoint_path}")
    return model
