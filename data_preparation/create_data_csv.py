# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import csv
from pathlib import Path

import cv2


def create_data_csv():
    """
    STEP1
    capture_movieディレクトリ内の動画ファイルを走査し、
    各動画のファイル名と画面サイズをmovies.csvに保存する
    """
    # ディレクトリパス
    capture_movie_dir = Path("capture_movie")
    capture_data_dir = Path("capture_data")
    output_csv = "movies.csv"

    # 結果を格納するリスト
    results = []

    # capture_movieディレクトリ内の動画ファイルを取得
    video_files = sorted(capture_movie_dir.glob("*.mp4"))

    print(f"Found {len(video_files)} video files")

    for video_path in video_files:
        filename = video_path.stem  # 拡張子なしのファイル名
        csv_path = capture_data_dir / f"{filename}.csv"

        # 対応するCSVファイルが存在するか確認
        if not csv_path.exists():
            print(f"Warning: CSV file not found for {filename}")
            continue

        # 動画を開いて画面サイズを取得
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        # 結果を追加
        results.append({
            'filename': filename,
            'width': width,
            'height': height
        })

        print(f"Processed: {filename} - {width}x{height}")

    # CSVファイルに書き出し
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'width', 'height'])

        for result in results:
            writer.writerow([result['filename'], result['width'], result['height']])

    print(f"\nCreated {output_csv} with {len(results)} entries")

    # 画面サイズの統計を表示
    if results:
        unique_sizes = set((r['width'], r['height']) for r in results)
        print(f"\nUnique screen sizes found:")
        for size in unique_sizes:
            count = sum(1 for r in results if (r['width'], r['height']) == size)
            print(f"  {size[0]}x{size[1]}: {count} files")


if __name__ == "__main__":
    create_data_csv()
