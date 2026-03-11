"""
evaluate/evaluator.py
=====================
학습된 모델 평가 및 예측 함수
"""

import os
import cv2
import torch
import numpy as np
from config.config import cfg
from data.dataset import read_image, read_mask, binarize_mask
from data.transforms import get_transforms
from utils.metrics import calc_iou, calc_dice, calc_pixel_accuracy
from utils.visualize import plot_predictions, plot_overlay


def predict_single(model, image_path: str) -> np.ndarray:
    """
    단일 이미지 예측

    Args:
        model     : 로드된 U-Net 모델
        image_path: 이미지 파일 경로
    Returns:
        예측 마스크 (H, W), 0~1 float32
    """
    transform = get_transforms("val")
    model.eval()

    # 읽기 + 전처리
    img = read_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dummy_mask = np.zeros(img.shape[:2], dtype=np.float32)
    aug = transform(image=img, mask=dummy_mask)
    inp = aug["image"].unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

    return pred


def evaluate_dataset(model, val_loader) -> dict:
    """
    검증 데이터셋 전체 평가

    Args:
        model     : 로드된 U-Net 모델
        val_loader: 검증 DataLoader
    Returns:
        {"iou", "dice", "pixel_acc"} 딕셔너리
    """
    model.eval()
    total_iou  = 0
    total_dice = 0
    total_acc  = 0
    n = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(cfg.DEVICE)
            masks = masks.to(cfg.DEVICE)
            preds = model(imgs)

            total_iou  += calc_iou(preds, masks)
            total_dice += calc_dice(preds, masks)
            total_acc  += calc_pixel_accuracy(preds, masks)
            n += 1

    results = {
        "iou"      : total_iou  / n,
        "dice"     : total_dice / n,
        "pixel_acc": total_acc  / n,
    }

    print("\n===== 평가 결과 =====")
    print(f"IoU        : {results['iou']:.4f}")
    print(f"Dice Score : {results['dice']:.4f}")
    print(f"Pixel Acc  : {results['pixel_acc']:.4f}")

    return results


def visualize_results(
    model,
    image_dir: str = None,
    mask_dir: str  = None,
    n: int = 4,
    save_path: str = "predictions.png"
):
    """
    모델 예측 결과 시각화 (원본 / 정답 / 예측 나란히)

    Args:
        model     : 로드된 U-Net 모델
        image_dir : 이미지 폴더
        mask_dir  : 마스크 폴더
        n         : 시각화할 샘플 수
        save_path : 저장 경로
    """
    image_dir = image_dir or cfg.IMAGE_DIR
    mask_dir  = mask_dir  or cfg.MASK_DIR
    transform = get_transforms("val")
    model.eval()

    img_files  = sorted(os.listdir(image_dir))[:n]
    mask_files = sorted(os.listdir(mask_dir))[:n]

    images, masks, preds = [], [], []

    for img_f, mask_f in zip(img_files, mask_files):
        # 원본 이미지
        img = read_image(os.path.join(image_dir, img_f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 정답 마스크
        mask = read_mask(os.path.join(mask_dir, mask_f))
        mask_bin = binarize_mask(mask)

        # 추론
        aug = transform(image=img, mask=mask_bin)
        inp = aug["image"].unsqueeze(0).to(cfg.DEVICE)
        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

        images.append(img)
        masks.append(mask_bin)
        preds.append(pred)

    plot_predictions(images, masks, preds, save_path=save_path, n=n)
