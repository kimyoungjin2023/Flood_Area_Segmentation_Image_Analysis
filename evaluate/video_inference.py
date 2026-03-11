"""
evaluate/video_inference.py
===========================
학습된 모델로 영상 테스트

지원:
  - mp4/avi 영상 파일
  - 웹캠 실시간
  - 이미지 파일 여러 장
"""

import os
import cv2
import torch
import numpy as np
from config.config import cfg
from models.unet import load_model
from data.transforms import get_transforms


# ============================================================
# ⚙️ 설정
# ============================================================
CHECKPOINT  = "./checkpoints/best_model.pth"   # 모델 경로
THRESHOLD   = 0.5                               # 예측 이진화 기준
OVERLAY_COLOR = (0, 100, 255)                   # 홍수 표시 색 (BGR, 주황)
OVERLAY_ALPHA = 0.5                             # 오버레이 투명도


# ============================================================
# 유틸 함수
# ============================================================
def load_inference_model():
    """모델 로드"""
    model = load_model(CHECKPOINT)
    model.eval()
    print(f"모델 로드 완료: {CHECKPOINT}")
    return model


def preprocess_frame(frame: np.ndarray):
    """
    프레임 전처리 (BGR → RGB → Tensor)

    Args:
        frame: OpenCV BGR 프레임
    Returns:
        (tensor, original_size) 튜플
    """
    transform = get_transforms("val")
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 더미 마스크로 transform 적용
    dummy_mask = np.zeros((h, w), dtype=np.float32)
    aug = transform(image=rgb, mask=dummy_mask)
    tensor = aug["image"].unsqueeze(0).to(cfg.DEVICE)

    return tensor, (h, w)


def predict_frame(model, frame: np.ndarray) -> np.ndarray:
    """
    단일 프레임 예측

    Args:
        model: 로드된 모델
        frame: BGR 프레임
    Returns:
        예측 마스크 (원본 해상도, 0~1 float32)
    """
    tensor, (orig_h, orig_w) = preprocess_frame(frame)

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
        pred = pred.squeeze().cpu().numpy()   # (512, 512)

    # 원본 해상도로 복원
    pred_resized = cv2.resize(pred, (orig_w, orig_h))
    return pred_resized


def draw_overlay(frame: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    원본 프레임 위에 홍수 영역 오버레이

    Args:
        frame    : BGR 원본 프레임
        pred_mask: 예측 마스크 (0~1)
    Returns:
        오버레이된 BGR 프레임
    """
    result = frame.copy()
    flood_region = pred_mask > THRESHOLD

    # 홍수 영역 색상 오버레이
    overlay = frame.copy()
    overlay[flood_region] = OVERLAY_COLOR
    result = cv2.addWeighted(frame, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)

    # 침수 면적 비율 계산
    flood_ratio = flood_region.sum() / flood_region.size * 100

    # 텍스트 표시
    cv2.putText(
        result,
        f"Flood: {flood_ratio:.1f}%",
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        (0, 255, 255), 3
    )

    # 침수 비율에 따라 경고 표시
    if flood_ratio > 30:
        cv2.putText(
            result, "!! FLOOD DETECTED !!",
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2
        )

    return result


# ============================================================
# MODE 1. 영상 파일 테스트
# ============================================================
def run_video(
    video_path: str,
    output_path: str = "output_video.mp4",
    show: bool = True
):
    """
    mp4/avi 영상 파일로 테스트

    Args:
        video_path : 입력 영상 경로
        output_path: 결과 영상 저장 경로
        show       : 화면 실시간 출력 여부
    """
    model = load_inference_model()
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"영상을 열 수 없어요: {video_path}")
        return

    # 영상 정보
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"입력 영상: {width}x{height} @ {fps:.1f}fps, 총 {total}프레임")

    # 출력 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 예측 + 오버레이
        pred_mask = predict_frame(model, frame)
        result    = draw_overlay(frame, pred_mask)

        # 프레임 번호 표시
        cv2.putText(
            result, f"{frame_idx}/{total}",
            (width - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )

        out.write(result)

        if show:
            cv2.imshow("Flood Detection", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
                print("사용자가 종료했습니다.")
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"처리 중: {frame_idx}/{total} ({frame_idx/total*100:.1f}%)")

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n완료! 결과 저장: {output_path}")


# ============================================================
# MODE 2. 웹캠 실시간 테스트
# ============================================================
def run_webcam(camera_id: int = 0):
    """
    웹캠 실시간 테스트

    Args:
        camera_id: 카메라 번호 (기본 0 = 내장 웹캠)
                   RTSP 스트림: "rtsp://아이피:554/stream"
    """
    model = load_inference_model()
    cap   = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"카메라를 열 수 없어요: {camera_id}")
        return

    print("웹캠 시작! 'q' 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_mask = predict_frame(model, frame)
        result    = draw_overlay(frame, pred_mask)

        cv2.imshow("Flood Detection (실시간)", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("종료")


# ============================================================
# MODE 3. 이미지 여러 장 배치 테스트
# ============================================================
def run_images(
    image_dir: str,
    output_dir: str = "output_images",
    extensions: tuple = (".jpg", ".jpeg", ".png")
):
    """
    폴더 내 이미지 전체 테스트

    Args:
        image_dir : 입력 이미지 폴더
        output_dir: 결과 저장 폴더
        extensions: 처리할 확장자
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_inference_model()

    files = [
        f for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(extensions)
    ]
    print(f"처리할 이미지: {len(files)}장")

    for i, fname in enumerate(files):
        img_path = os.path.join(image_dir, fname)

        # 읽기
        frame = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if frame is None:
            print(f"읽기 실패: {fname}")
            continue

        # 예측 + 오버레이
        pred_mask = predict_frame(model, frame)
        result    = draw_overlay(frame, pred_mask)

        # 저장 (원본 / 마스크 / 오버레이 나란히)
        h, w = frame.shape[:2]
        mask_vis = (pred_mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, mask_bgr, result])  # 3장 가로로 합치기

        out_path = os.path.join(output_dir, fname)
        cv2.imencode(".jpg", combined)[1].tofile(out_path)

        if (i + 1) % 10 == 0:
            print(f"진행: {i+1}/{len(files)}")

    print(f"\n완료! 결과 저장: {output_dir}/")
    print("저장 형식: [원본 | 마스크 | 오버레이] 가로로 합친 이미지")


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":

    # ── 원하는 모드 주석 해제해서 실행 ──────────────────────

    # 모드 1: 영상 파일
    run_video(
        video_path  = "./dataset/video.mp4",        # ← 영상 파일 경로
        output_path = "./dataset_result/datoutput_video.mp4",
        show        = True               # False면 화면 출력 없이 저장만
    )

    # 모드 2: 웹캠 실시간
    # run_webcam(camera_id=0)

    # 모드 3: 이미지 여러 장
    # run_images(
    #     image_dir  = "./test_images",
    #     output_dir = "./output_images"
    # )
