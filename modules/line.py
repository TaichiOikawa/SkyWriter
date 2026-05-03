# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import cv2
import numpy as np


class Line():
    """描画データ、1本線を表すクラス"""

    def __init__(self, x: int, y: int) -> None:
        self.points = []
        self.points.append((x, y))

    def write(self, x: int, y: int):
        """線を引く"""
        self.points.append((x, y))

    def to_board(self, board: np.ndarray):
        """boardに描画 (pointsが2点以上ある場合のみ描画)"""
        if len(self.points) < 2:
            return board
        for i in range(1, len(self.points)):
            cv2.line(board, self.points[i - 1], self.points[i], (0, 0, 0), 5)
        return board
