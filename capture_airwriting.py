# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import os
import csv
import argparse
import datetime

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.character import CharacterExtractor


# MediaPipe モジュール
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# しきい値
DISTANCE_THRESHOLD = 30  # 距離のしきい値
GOOGLE_CREDENTIALS_PATH = "./gcp-my_first_project-auth.json"

# ログ保存用ディレクトリ
os.makedirs("capture_movie", exist_ok=True)
os.makedirs("capture_data", exist_ok=True)
os.makedirs("capture_image", exist_ok=True)

recording_number = 0


def putText_japanese(img, text, point, size, color):
    # Notoフォントとする
    font = ImageFont.truetype('C:\\Windows\\Fonts\\NotoSansJP-VF.ttf', size)

    # imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)

    # drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)

    # テキスト描画
    draw.text(point, text, fill=color, font=font)

    # PILからndarrayに変換して返す
    return np.array(img_pil)


def main(movie_path: str | None):
    global recording_number
    if movie_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        print("動画ファイルが開けません。")
        return

    session_name = f"{os.path.splitext(os.path.basename(movie_path))[0]}_capture" if movie_path else f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = cv2.VideoWriter(f'capture_movie/{session_name}.mp4', cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 30.0, (frame_width, frame_height))
    csv_file = open(f'capture_data/{session_name}.csv', "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "thumb_x", "thumb_y", "index_x", "index_y", "distance", "dx", "dy", "timestamp", "is_writing"])

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:

        board = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        before_point_x = None
        before_point_y = None

        csv_file = None
        frame_count = 0
        prev_thumb = None
        is_writing = True

        ocr = CharacterExtractor(GOOGLE_CREDENTIALS_PATH)
        text = None

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 人差し指と親指の座標格納
            index_finger_tips = []
            thumb_tips = []
            distance = None
            dx, dy = 0, 0

            hands_results = hands.process(rgb)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Right":
                        index_finger_tip = (
                            int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                            int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                        )
                        thumb_tip = (
                            int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
                            int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                        )
                        index_finger_tips.append(index_finger_tip)
                        thumb_tips.append(thumb_tip)

                        # 距離計算
                        distance = ((thumb_tip[0] - index_finger_tip[0]) ** 2 + (thumb_tip[1] - index_finger_tip[1]) ** 2) ** 0.5
                        if prev_thumb:
                            dx = thumb_tip[0] - prev_thumb[0]
                            dy = thumb_tip[1] - prev_thumb[1]
                        prev_thumb = thumb_tip

                    # ランドマーク描画
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                    )

                    # 人差し指と親指を表示
                    for tip in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.THUMB_TIP]:
                        lm = hand_landmarks.landmark[tip]
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                        cv2.putText(frame, f'X:{x} Y:{y}', (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # 書く動作
                        if tip == mp_hands.HandLandmark.THUMB_TIP and thumb_tips and index_finger_tips:
                            if distance is not None:
                                cv2.putText(frame, f'Dist:{int(distance)}', (50, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                if distance < DISTANCE_THRESHOLD:
                                    cv2.putText(frame, 'Pinch Detected', (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                    if before_point_x is not None and before_point_y is not None:
                                        cv2.line(board, (int(before_point_x), int(before_point_y)), thumb_tips[0], (0, 0, 0), 5)

                    before_point_x, before_point_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])

            if writer and thumb_tips and index_finger_tips:
                thumb_x, thumb_y = thumb_tips[0]
                index_x, index_y = index_finger_tips[0]
                writer.writerow([
                    frame_count,
                    thumb_x, thumb_y,
                    index_x, index_y,
                    round(distance if distance else 0, 2),
                    dx, dy,
                    cv2.getTickCount() / cv2.getTickFrequency(),
                    1 if is_writing else 0
                ])
            frame_count += 1

            # 合成表示
            frame = cv2.addWeighted(frame, 0.5, board, 0.5, 0)
            writing_text = "Writing" if is_writing else "Idle"
            cv2.putText(frame, writing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if is_writing else (200, 200, 200), 2)

            frame = putText_japanese(frame, f'OCR Text: {text if text else ""}', (10, frame.shape[0] - 100), 64, (255, 0, 0))

            cv2.imshow('MediaPipe Hands', frame)

            video.write(frame)

            pushed_key = cv2.waitKey(1) & 0xFF

            # 保存/リセット/終了
            if pushed_key == ord('w'):
                cv2.imwrite('board.png', board)
                ocr.image = None
                ocr.add_character(board)
                text = ocr.extract_text()
            if pushed_key == ord('r'):
                board = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
            if pushed_key == ord('q'):
                break

            if pushed_key == ord(' '):
                if is_writing:
                    is_writing = False
                    cv2.imwrite(f'capture_image/{session_name}_{frame_count}.png', board)
                    board = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                else:
                    is_writing = True

    video.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie", type=str, default=None, help="Path to the movie file to process.")
    args = parser.parse_args()
    main(movie_path=args.movie)
