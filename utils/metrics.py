"""
utils/metrics.py
================
Segmentation 성능 지표 계산
"""

import torch
import numpy as np


def calc_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    IoU (Intersection over Union) 계산
    - 예측 마스크와 정답 마스크가 얼마나 겹치는지
    - 1.0 = 완벽, 0.7 이상이면 실무에서 좋은 성능

    Args:
        pred     : 모델 출력 (logits, sigmoid 전)
        target   : 정답 마스크 (0 또는 1)
        threshold: 이진화 기준값
    """
    pred_bin     = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    union        = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou          = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def calc_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Dice Score (F1 Score) 계산
    - IoU와 비슷하지만 작은 영역에 더 관대
    - 의료/홍수 분야에서 자주 사용

    Args:
        pred     : 모델 출력 (logits)
        target   : 정답 마스크
        threshold: 이진화 기준값
    """
    pred_bin     = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    dice         = (2 * intersection + 1e-6) / (
        pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
    )
    return dice.mean().item()


def calc_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    픽셀 정확도 계산
    - 전체 픽셀 중 올바르게 분류된 비율
    - 클래스 균형이 좋을 때 유용

    Args:
        pred     : 모델 출력 (logits)
        target   : 정답 마스크
        threshold: 이진화 기준값
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    correct  = (pred_bin == target).float().sum()
    total    = torch.numel(target)
    return (correct / total).item()


class MetricTracker:
    """
    에포크별 지표 누적 및 평균 계산
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums  = {}
        self._counts = {}

    def update(self, **kwargs):
        """지표값 추가"""
        for key, val in kwargs.items():
            if key not in self._sums:
                self._sums[key]   = 0.0
                self._counts[key] = 0
            self._sums[key]   += val
            self._counts[key] += 1

    def avg(self, key: str) -> float:
        """평균 반환"""
        if self._counts.get(key, 0) == 0:
            return 0.0
        return self._sums[key] / self._counts[key]

    def summary(self) -> dict:
        """모든 지표 평균 딕셔너리 반환"""
        return {k: self.avg(k) for k in self._sums}
