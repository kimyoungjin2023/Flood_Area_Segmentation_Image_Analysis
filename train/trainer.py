"""
train/trainer.py
================
학습 / 검증 루프 함수
"""

import os
import torch
from tqdm import tqdm
from config.config import cfg
from utils.metrics import MetricTracker, calc_iou, calc_dice


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device=None
) -> dict:
    """
    1 에포크 학습

    Args:
        model    : U-Net 모델
        loader   : Train DataLoader
        optimizer: 옵티마이저
        criterion: 손실 함수
        device   : 학습 디바이스
    Returns:
        {"loss", "iou", "dice"} 평균값 딕셔너리
    """
    device  = device or cfg.DEVICE
    tracker = MetricTracker()
    model.train()

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        tracker.update(
            loss = loss.item(),
            iou  = calc_iou(preds, masks),
            dice = calc_dice(preds, masks),
        )

    return tracker.summary()


def validate(
    model,
    loader,
    criterion,
    device=None
) -> dict:
    """
    검증 루프

    Args:
        model    : U-Net 모델
        loader   : Val DataLoader
        criterion: 손실 함수
        device   : 디바이스
    Returns:
        {"loss", "iou", "dice"} 평균값 딕셔너리
    """
    device  = device or cfg.DEVICE
    tracker = MetricTracker()
    model.eval()

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss  = criterion(preds, masks)

            tracker.update(
                loss = loss.item(),
                iou  = calc_iou(preds, masks),
                dice = calc_dice(preds, masks),
            )

    return tracker.summary()


def save_checkpoint(model, path: str):
    """모델 가중치 저장"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  체크포인트 저장: {path}")


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs: int   = None,
    lr: float     = None,
    save_path: str = None,
) -> dict:
    """
    전체 학습 실행 함수

    Args:
        model       : U-Net 모델
        train_loader: 학습 DataLoader
        val_loader  : 검증 DataLoader
        criterion   : 손실 함수
        epochs      : 학습 에포크 수
        lr          : 학습률
        save_path   : Best 모델 저장 경로
    Returns:
        history 딕셔너리
    """
    epochs    = epochs    or cfg.EPOCHS
    lr        = lr        or cfg.LR
    save_path = save_path or os.path.join(cfg.SAVE_DIR, "best_model.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    history  = {
        "train_loss": [], "val_loss": [],
        "train_iou" : [], "val_iou" : [],
        "train_dice": [], "val_dice": [],
    }
    best_iou = 0.0

    print(f"\n{'='*55}")
    print(f"  학습 시작 | Epochs: {epochs} | Device: {cfg.DEVICE}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics   = validate(model, val_loader, criterion)
        scheduler.step(val_metrics["iou"])

        # 기록
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"IoU: {train_metrics['iou']:.4f}/{val_metrics['iou']:.4f} | "
            f"Dice: {train_metrics['dice']:.4f}/{val_metrics['dice']:.4f}"
        )

        # Best 모델 저장
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(model, save_path)
            print(f"  ✅ Best 모델 갱신! Val IoU: {best_iou:.4f}")

    print(f"\n학습 완료! Best Val IoU: {best_iou:.4f}")
    return history
