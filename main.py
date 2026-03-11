"""
main.py
=======
프로젝트 진입점
학습 / 평가 / 예측을 한 곳에서 실행
"""

import torch
import numpy as np

from config.config import cfg
from data.dataset import build_dataloaders
from models.unet import build_model, load_model
from utils.losses import get_loss
from utils.visualize import plot_training_history
from train.trainer import run_training
from evaluate.evaluator import evaluate_dataset, visualize_results


def main(mode: str = "train"):
    """
    Args:
        mode: "train" | "evaluate" | "predict"
    """
    # 재현성 고정
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print(f"디바이스: {cfg.DEVICE}")
    print(f"모드    : {mode}\n")

    # ── 공통: 데이터 로더 ────────────────────────────────
    train_loader, val_loader = build_dataloaders()

    if mode == "train":
        # ── 학습 ─────────────────────────────────────────
        model     = build_model()
        criterion = get_loss("dice_bce")

        history = run_training(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            criterion    = criterion,
        )

        # 학습 곡선 저장
        plot_training_history(history)

        # 학습 후 바로 예측 결과 시각화
        best_model = load_model(f"{cfg.SAVE_DIR}/best_model.pth")
        visualize_results(best_model, n=4)

    elif mode == "evaluate":
        # ── 평가 ─────────────────────────────────────────
        model   = load_model(f"{cfg.SAVE_DIR}/best_model.pth")
        results = evaluate_dataset(model, val_loader)

    elif mode == "predict":
        # ── 단일 이미지 예측 ──────────────────────────────
        from evaluate.evaluator import predict_single
        import matplotlib.pyplot as plt

        model      = load_model(f"{cfg.SAVE_DIR}/best_model.pth")
        image_path = "./dataset/image/0.jpg"   # ← 예측할 이미지 경로
        pred_mask  = predict_single(model, image_path)

        plt.figure(figsize=(6, 5))
        plt.imshow(pred_mask, cmap="gray")
        plt.title("예측 마스크")
        plt.axis("off")
        plt.savefig("single_prediction.png")
        plt.show()
        print("저장 완료: single_prediction.png")

    else:
        raise ValueError(f"지원하지 않는 mode: {mode} | train / evaluate / predict 중 선택")


if __name__ == "__main__":
    # mode 변경: "train" | "evaluate" | "predict"
    main(mode="train")
