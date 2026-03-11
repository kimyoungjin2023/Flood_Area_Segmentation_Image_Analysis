"""
utils/visualize.py
==================
학습 곡선 및 예측 결과 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from config.config import cfg

# 한글 폰트 설정
plt.rcParams['font.family']         = cfg.FONT_FAMILY
plt.rcParams['axes.unicode_minus']  = False


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    """
    학습/검증 Loss & IoU 곡선 시각화

    Args:
        history  : {"train_loss", "val_loss", "train_iou", "val_iou"} 딕셔너리
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 곡선
    axes[0].plot(history["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#3498db")
    axes[0].set_title("Loss 곡선")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU 곡선
    axes[1].plot(history["train_iou"], label="Train IoU", color="#e74c3c")
    axes[1].plot(history["val_iou"],   label="Val IoU",   color="#3498db")
    axes[1].set_title("IoU 곡선")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"학습 곡선 저장: {save_path}")


def plot_predictions(
    images: list,
    masks: list,
    preds: list,
    save_path: str = "predictions.png",
    n: int = 4
):
    """
    원본 이미지 / 정답 마스크 / 예측 마스크 나란히 시각화

    Args:
        images   : RGB 이미지 리스트 (numpy)
        masks    : 정답 마스크 리스트 (numpy, 0~1)
        preds    : 예측 마스크 리스트 (numpy, 0~1)
        save_path: 저장 경로
        n        : 시각화할 샘플 수
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = [axes]

    titles = ["원본 이미지", "정답 마스크", "예측 마스크"]
    for i in range(n):
        for j, (data, title) in enumerate(zip(
            [images[i], masks[i], preds[i]], titles
        )):
            axes[i][j].imshow(data, cmap="gray" if j > 0 else None)
            axes[i][j].set_title(title)
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"예측 결과 저장: {save_path}")


def plot_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
    save_path: str = "overlay.png"
):
    """
    원본 이미지 위에 예측 마스크를 반투명하게 오버레이

    Args:
        image    : RGB 이미지 (H, W, 3)
        pred_mask: 예측 마스크 (H, W), 0~1
        alpha    : 오버레이 투명도
        save_path: 저장 경로
    """
    overlay = image.copy()
    mask_region = pred_mask > 0.5

    # 홍수 영역을 파란색으로 표시
    overlay[mask_region] = (
        overlay[mask_region] * (1 - alpha) +
        np.array([0, 100, 255]) * alpha
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("원본 이미지")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("홍수 영역 오버레이 (파란색)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"오버레이 저장: {save_path}")
