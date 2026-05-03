# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import csv
from pathlib import Path

import numpy as np


def add_features():
    """
    STEP.3
    normalized_capture_data内のCSVファイルに新しい特徴量を追加する
    - velocity: sqrt(dx^2 + dy^2) - 速度の大きさ
    - acceleration: 前フレームとの速度変化
    - angle: atan2(dy, dx) - 移動方向（ラジアン）
    - distance_change: 距離の変化率
    """
    input_dir = Path("normalized_capture_data")
    output_dir = Path("training_data")

    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True)

    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        return

    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    processed_count = 0

    for csv_path in csv_files:
        rows = []
        prev_velocity = 0.0
        prev_distance = 0.0

        # CSVファイルを読み込む
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            original_fieldnames = reader.fieldnames

            for i, row in enumerate(reader):
                # 既存のデータをコピー
                enhanced_row = row.copy()

                # dx, dy から特徴量を計算
                dx = float(row['dx'])
                dy = float(row['dy'])
                distance = float(row['distance'])

                # 1. 速度の大きさ
                velocity = np.sqrt(dx**2 + dy**2)
                enhanced_row['velocity'] = velocity

                # 2. 加速度（前フレームとの速度変化）
                if i == 0:
                    acceleration = 0.0
                else:
                    acceleration = velocity - prev_velocity
                enhanced_row['acceleration'] = acceleration

                # 3. 角度（移動方向）
                # atan2(dy, dx) はラジアンで返される（-π から π の範囲）
                angle = np.arctan2(dy, dx)
                enhanced_row['angle'] = angle

                # 4. 距離の変化率
                if i == 0:
                    distance_change = 0.0
                else:
                    distance_change = distance - prev_distance
                enhanced_row['distance_change'] = distance_change

                rows.append(enhanced_row)

                # 次のループのために保存
                prev_velocity = velocity
                prev_distance = distance

        # 新しいフィールド名を定義
        new_fieldnames = list(original_fieldnames) + ['velocity', 'acceleration', 'angle', 'distance_change']

        # 拡張したデータを保存
        output_path = output_dir / csv_path.name
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        processed_count += 1
        print(f"Processed: {csv_path.name} - {len(rows)} rows, added 4 features")

    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count} files")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nAdded features:")
    print(f"  - velocity: sqrt(dx^2 + dy^2)")
    print(f"  - acceleration: velocity change from previous frame")
    print(f"  - angle: atan2(dy, dx) in radians")
    print(f"  - distance_change: distance change from previous frame")


if __name__ == "__main__":
    add_features()
