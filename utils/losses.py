"""
utils/losses.py
===============
손실 함수 정의
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss
    - 홍수처럼 영역이 불균일한 Segmentation에 적합
    - IoU와 비슷하지만 경계에 더 민감
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        dice = 1 - (2 * intersection + self.smooth) / (
            pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        return dice.mean()


class DiceBCELoss(nn.Module):
    """
    Dice Loss + BCE Loss 결합
    - BCE: 픽셀 단위 분류 정확도
    - Dice: 영역 겹침 최대화
    - 둘을 합치면 더 안정적으로 학습됨 (추천)
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    - 클래스 불균형이 심할 때 사용
    - 어려운 샘플에 더 집중해서 학습
    - 현재 데이터셋은 균형 양호라 DiceBCE 추천
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def get_loss(name: str = "dice_bce") -> nn.Module:
    """
    이름으로 손실 함수 반환

    Args:
        name: "dice_bce" | "dice" | "bce" | "focal"
    """
    losses = {
        "dice_bce": DiceBCELoss(),
        "dice"    : DiceLoss(),
        "bce"     : nn.BCEWithLogitsLoss(),
        "focal"   : FocalLoss(),
    }
    if name not in losses:
        raise ValueError(f"지원하지 않는 손실 함수: {name}\n선택지: {list(losses.keys())}")
    print(f"손실 함수: {name}")
    return losses[name]
