# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import os
from datetime import datetime
import argparse

import cv2
import mediapipe as mp
import numpy as np
import concurrent.futures
import torch

from modules.line import Line  # Line class
from modules.character import CharacterExtractor  # OCR module

from train_separator import MotionSeparator  # Trained model


# ======================
# MediaPipe Initialization
# ======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ======================
# Settings
# ======================
MODEL_INFO_PATH = "models.json"
DISTANCE_THRESHOLD = 25  # ピンチ判定閾値
SEQ_LEN = 30             # リアルタイム推論用シーケンス長
NON_WRITING_FRAMES_THRESHOLD = 10  # 書いていないフレーム連続閾値
# LSTM確率のヒステリシス閾値（立ち上がり/立ち下がりで異なる閾値を使いチャタリング防止）
PRED_ON_THRESH = 0.85
PRED_OFF_THRESH = 0.5
GOOGLE_CREDENTIALS_PATH = "./gcp-my_first_project-auth.json"  # Google Cloud認証情報パス

# 入力特徴量
INPUT_FEATURES = ["thumb_x", "thumb_y", "index_x", "index_y", "distance", "dx", "dy", "velocity", "acceleration", "angle", "distance_change"]

IMAGE_DIR = "separator_images/"
os.makedirs(IMAGE_DIR, exist_ok=True)

session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ======================
# Main
# ======================
def main(model_path: str, camera_id: int = 0, no_extract: bool = False, save_image: bool = False, from_video: str | None = None, record_video: str | None = None):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotionSeparator(input_dim=len(INPUT_FEATURES))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # バッファをnumpy配列で管理（効率的なテンソル変換のため）
    buffer = np.zeros((SEQ_LEN, len(INPUT_FEATURES)), dtype=np.float32)
    buffer_count = 0  # バッファに追加されたデータの数

    def predict_boundary():
        """推測実行"""
        if buffer_count < SEQ_LEN:
            return None
        # numpy配列から直接テンソルを作成
        x = torch.from_numpy(buffer).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).item()
        return prob

    def add_to_buffer(features):
        """numpyバッファに特徴量を追加（循環バッファ）"""
        nonlocal buffer_count
        idx = buffer_count % SEQ_LEN
        buffer[idx] = features
        buffer_count += 1

    def extract_features(thumb, index, distance, dx, dy, frame_width, frame_height, prev_velocity=0.0, prev_distance=0.0):
        """正規化処理"""
        diagonal = (frame_width**2 + frame_height**2)**0.5  # 画面の対角線長

        # 正規化された dx, dy
        norm_dx = dx / frame_width
        norm_dy = dy / frame_height

        # 新しい特徴量の計算
        # 1. 速度の大きさ (正規化された値を使用)
        velocity = np.sqrt(norm_dx**2 + norm_dy**2)

        # 2. 加速度 (前フレームとの速度変化)
        acceleration = velocity - prev_velocity

        # 3. 角度 (移動方向、ラジアン)
        angle = np.arctan2(norm_dy, norm_dx)

        # 4. 距離の変化率 (正規化された距離を使用)
        norm_distance = distance / diagonal
        distance_change = norm_distance - prev_distance

        param_map = {
            "thumb_x": thumb[0] / frame_width,
            "thumb_y": thumb[1] / frame_height,
            "index_x": index[0] / frame_width,
            "index_y": index[1] / frame_height,
            "distance": norm_distance,
            "dx": norm_dx,
            "dy": norm_dy,
            "velocity": velocity,
            "acceleration": acceleration,
            "angle": angle,
            "distance_change": distance_change
        }
        feat = [param_map[p] for p in INPUT_FEATURES]

        # NaN/Inf チェック
        if any(np.isnan(f) or np.isinf(f) for f in feat):
            print(f"[WARNING] NaN or Inf detected in features: {feat}")
            # ゼロで置き換え
            feat = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in feat]

        return feat

    if from_video is None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Webカメラが開けません")
            return
    else:
        cap = cv2.VideoCapture(from_video)
        if not cap.isOpened():
            print("ビデオファイルが開けません")
            return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        board = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        before_point_x = before_point_y = None
        prev_thumb = None
        before_boundary_eff = 0  # ヒステリシス&ピンチゲート後の有効な境界状態
        boundary_pred_eff = 0  # 初期値を0に変更（バッファ不足時は書いていないと判定）
        non_writing_count = 0
        lines = []
        is_pen_down = False
        before_is_pen_down = False
        boundary_prob = None
        hand_detected_once = False  # 手が一度でも検出されたかどうか]

        # 新しい特徴量のために前フレーム情報を保持
        prev_velocity = 0.0
        prev_distance = 0.0

        # OCR を非同期で実行するための Executor
        ocr_local = CharacterExtractor(GOOGLE_CREDENTIALS_PATH)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        ocr_futures = []  # 保留中の Future を保持

        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        video = None
        if record_video:
            video = cv2.VideoWriter(record_video, cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 30.0, (frame_width, frame_height))

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            index_finger_tips = []
            thumb_tips = []
            distance = dx = dy = 0

            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Right":
                        ix = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                        iy = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                        tx = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                        ty = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                        index_finger_tips.append((ix, iy))
                        thumb_tips.append((tx, ty))
                        distance = ((tx - ix)**2 + (ty - iy)**2)**0.5
                        if prev_thumb:
                            dx = tx - prev_thumb[0]
                            dy = ty - prev_thumb[1]
                        prev_thumb = (tx, ty)

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

            # 書く動作判定 & board描画
            boundary_prob = None  # LSTMの生確率
            if thumb_tips and index_finger_tips:
                hand_detected_once = True  # 手が検出された
                if distance < DISTANCE_THRESHOLD:
                    if before_point_x is not None and before_point_y is not None:
                        if is_pen_down is False:
                            lines.append(Line(thumb_tips[0][0], thumb_tips[0][1]))
                            is_pen_down = True
                        else:
                            lines[-1].write(thumb_tips[0][0], thumb_tips[0][1])
                else:
                    is_pen_down = False
                before_point_x, before_point_y = thumb_tips[0]

                # 特徴量バッファ（正規化して追加）
                feat = extract_features(
                    thumb_tips[0], index_finger_tips[0], distance, dx, dy,
                    frame_width, frame_height, prev_velocity, prev_distance
                )
                add_to_buffer(feat)
                boundary_prob = predict_boundary()

                # 次フレームのために現在の値を保存
                diagonal_norm = (frame_width**2 + frame_height**2)**0.5
                norm_dx = dx / frame_width
                norm_dy = dy / frame_height
                prev_velocity = np.sqrt(norm_dx**2 + norm_dy**2)
                prev_distance = distance / diagonal_norm
            else:
                # 手が検出されなかった場合もゼロパディングして時系列の連続性を保つ
                before_point_x = before_point_y = None
                prev_thumb = None
                # バッファをクリアせず、ゼロベクトルを追加
                zero_feat = np.zeros(len(INPUT_FEATURES), dtype=np.float32)
                add_to_buffer(zero_feat)
                is_pen_down = False
                non_writing_count += 1
                # 一度手が検出された後、検出されなくなったら0にする
                if hand_detected_once:
                    boundary_pred_eff = 0
                # 前フレーム情報もリセット
                prev_velocity = 0.0
                prev_distance = 0.0

            # ヒステリシス + ピンチゲーティングで有効境界を算出
            pinch_active = (distance < DISTANCE_THRESHOLD) and bool(thumb_tips) and bool(index_finger_tips)

            # LSTM確率値に基づいてヒステリシス判定
            if boundary_prob is not None:
                # 確率が低い閾値以下: 明確に"書いていない"
                is_clearly_not_writing = boundary_prob <= PRED_OFF_THRESH
                # 確率が高い閾値以上: 明確に"書いている"
                is_clearly_writing = boundary_prob >= PRED_ON_THRESH
                # 中間領域: 前の状態を維持（チャタリング防止）

                if is_clearly_not_writing:
                    boundary_pred_eff = 0
                elif is_clearly_writing:
                    boundary_pred_eff = 1
                else:
                    boundary_pred_eff = before_boundary_eff

            # --- LSTM推論による文字分割判定 ---
            # --- 書き始め(非書字→書字)の検出 ---

            # 状態遷移の検出
            lstm_state_changed = (before_boundary_eff == 0 and boundary_pred_eff == 1)
            pen_down_detected = (before_is_pen_down == False and is_pen_down == True)
            sufficient_pause = non_writing_count >= NON_WRITING_FRAMES_THRESHOLD

            # 書き始め条件: (LSTM遷移 または ペンダウン) かつ 十分な停止期間
            writing_start_detected = sufficient_pause and (lstm_state_changed or pen_down_detected)

            if writing_start_detected:
                print('--- LSTM Start Detected (書き始め) ---')
                # 書き始め時にボードをリセット（直前の線は残す）
                last_line = lines[-1] if (len(lines) > 0 and is_pen_down) else None
                board_copy = board.copy()

                if not no_extract:
                    # 非同期で OCR を実行するジョブを投げる
                    def _ocr_job(img):
                        try:
                            ocr_local.add_character(img)
                            return ocr_local.extract_text()
                        except Exception as e:
                            return f"OCR error: {e}"

                    future = executor.submit(_ocr_job, board_copy)
                    ocr_futures.append(future)
                else:
                    ocr_local.add_character(board_copy)

                if save_image:
                    try:
                        ocr_local.save_image(f"{IMAGE_DIR}/{session_name}.png")
                    except Exception as e:
                        print(f"[ERROR] Failed to save image: {e}")

                board = np.ones_like(board) * 255
                if last_line is not None:
                    lines = [last_line]
                else:
                    lines = []

            # 次ループのために前回状態を更新
            before_boundary_eff = boundary_pred_eff
            before_is_pen_down = is_pen_down

            # non_writing_count の更新（書き始め判定の後に行う）
            if boundary_pred_eff == 0:
                non_writing_count += 1
            else:
                non_writing_count = 0

            # フレーム合成
            for line in lines:
                board = line.to_board(board)
            frame_disp = cv2.addWeighted(frame, 0.5, board, 0.5, 0)
            if len(buffer) >= SEQ_LEN and boundary_prob is not None:
                boundary_status = f"LSTM: {boundary_pred_eff} prob={boundary_prob:.2f} pinch={'Y' if pinch_active else 'N'}"
            else:
                boundary_status = f"LSTM: Buffering ({boundary_pred_eff})"
            cv2.putText(frame_disp, boundary_status, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if boundary_pred_eff == 1 else (0, 0, 255), 2)
            cv2.imshow("MediaPipe Hands", frame_disp)

            # 非同期 OCR の完了チェック（完了したら結果を表示）
            if not no_extract and ocr_futures:
                remaining = []
                for fut in ocr_futures:
                    if fut.done():
                        try:
                            text = fut.result()
                        except Exception as e:
                            text = f"OCR future error: {e}"
                        print("[Async OCR Result]:", text)
                    else:
                        remaining.append(fut)
                ocr_futures = remaining

            # 動画書き込み
            if video:
                video.write(frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                lines = []
                board = np.ones_like(board) * 255
                buffer[:] = 0  # numpy配列をゼロクリア
                buffer_count = 0  # バッファカウンターをリセット
                non_writing_count = 0

    # 最終文字のOCR処理
    if not no_extract:
        # 非同期で OCR を実行するジョブを投げる
        def _ocr_job(img):
            try:
                ocr_local.add_character(img)
                return ocr_local.extract_text()
            except Exception as e:
                return f"OCR error: {e}"

        future = executor.submit(_ocr_job, board)
        ocr_futures.append(future)
    else:
        ocr_local.add_character(board)

    if save_image:
        try:
            ocr_local.save_image(f"{IMAGE_DIR}/{session_name}.png")
        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")

    # シャットダウン: 保留中のOCRがあれば待たずに停止（必要なら wait=True に変更）
    try:
        executor.shutdown(wait=True)
    except NameError:
        pass

    if video:
        video.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="./model.pt", help="Path of model to use")
    parser.add_argument("--camera", "-c", type=int, default=0, help="ID of the camera to use")
    parser.add_argument("--no-extract", action="store_true", help="Do not perform OCR extraction")
    parser.add_argument("--save-image", "-s", action="store_true", help="Save the final board image")
    parser.add_argument("--from-video", "-v", type=str, default=None, help="Path to video file instead of camera")
    parser.add_argument("--record-video", "-r", type=str, help="Path to save the recorded video")
    args = parser.parse_args()
    main(model_path=args.model, camera_id=args.camera, no_extract=args.no_extract, save_image=args.save_image, from_video=args.from_video, record_video=args.record_video)
