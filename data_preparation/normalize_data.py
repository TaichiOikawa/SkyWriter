# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import csv
from pathlib import Path


def normalize_data():
    """
    STEP.2
    movies.csvを参照してcapture_data内のデータを正規化し、
    normalized_capture_dataディレクトリに保存する
    """
    # ディレクトリとファイルのパス
    data_csv = "movies.csv"
    capture_data_dir = Path("capture_data")
    output_dir = Path("normalized_capture_data")

    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True)

    # data.csvから画面サイズ情報を読み込む
    screen_sizes = {}
    with open(data_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            width = int(row['width'])
            height = int(row['height'])
            screen_sizes[filename] = (width, height)

    print(f"Loaded screen size info for {len(screen_sizes)} files")

    # 各CSVファイルを正規化
    normalized_count = 0
    skipped_count = 0

    for filename, (width, height) in screen_sizes.items():
        csv_path = capture_data_dir / f"{filename}.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            skipped_count += 1
            continue

        # 対角線長を計算
        diagonal = (width**2 + height**2)**0.5

        # CSVファイルを読み込んで正規化
        normalized_rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                # 各特徴量を正規化
                normalized_row = row.copy()

                # 座標の正規化（0-1の範囲に）
                normalized_row['thumb_x'] = float(row['thumb_x']) / width
                normalized_row['thumb_y'] = float(row['thumb_y']) / height
                normalized_row['index_x'] = float(row['index_x']) / width
                normalized_row['index_y'] = float(row['index_y']) / height

                # 距離の正規化（対角線長で割る）
                normalized_row['distance'] = float(row['distance']) / diagonal

                # 移動量の正規化
                normalized_row['dx'] = float(row['dx']) / width
                normalized_row['dy'] = float(row['dy']) / height

                # その他のフィールドはそのまま
                normalized_row['frame'] = row['frame']
                normalized_row['timestamp'] = row['timestamp']
                normalized_row['is_writing'] = row['is_writing']

                normalized_rows.append(normalized_row)

        # 正規化したデータを保存
        output_path = output_dir / f"{filename}.csv"
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized_rows)

        normalized_count += 1
        print(f"Normalized: {filename} ({width}x{height}) - {len(normalized_rows)} rows")

    print(f"\n=== Summary ===")
    print(f"Normalized: {normalized_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    normalize_data()
