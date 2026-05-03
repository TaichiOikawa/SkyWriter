# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import os
from typing import Optional, List

import cv2
import numpy as np

from google.cloud import vision
from google.oauth2 import service_account


class CharacterExtractor:
    def __init__(self, google_credentials_path: Optional[str] = None, client: Optional[vision.ImageAnnotatorClient] = None):
        self.image: Optional[np.ndarray] = None
        self._credentials_path = google_credentials_path
        self._client = client

    def _ensure_client(self) -> vision.ImageAnnotatorClient:
        if self._client is not None:
            return self._client
        if not self._credentials_path:
            raise ValueError("No Google credentials provided for OCR client.")
        creds = service_account.Credentials.from_service_account_file(self._credentials_path)
        self._client = vision.ImageAnnotatorClient(credentials=creds)
        return self._client

    @staticmethod
    def _hconcat_resize_min(im_list: List[np.ndarray], interpolation=cv2.INTER_CUBIC) -> np.ndarray:
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)

    @staticmethod
    def crop_horizontal_margin(input_image: np.ndarray, pad: int = 15, dark_thresh: int = 250) -> np.ndarray:
        if input_image is None:
            raise ValueError("input_image is None")
        if input_image.ndim == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_image.copy()

        mask_cols = np.any(gray < dark_thresh, axis=0)
        if not mask_cols.any():
            return input_image

        xs = np.where(mask_cols)[0]
        min_x, max_x = int(xs[0]), int(xs[-1])
        left = max(0, min_x - pad)
        right = min(input_image.shape[1] - 1, max_x + pad)
        return input_image[:, left:right + 1]

    def add_character(self, image: np.ndarray, pad: int = 15, dark_thresh: int = 250) -> None:
        if image is None:
            return
        img = self.crop_horizontal_margin(image, pad=pad, dark_thresh=dark_thresh)
        if self.image is None:
            self.image = img
        else:
            try:
                self.image = self._hconcat_resize_min([self.image, img])
            except Exception:
                h = min(self.image.shape[0], img.shape[0])
                a = cv2.resize(self.image, (int(self.image.shape[1] * h / self.image.shape[0]), h))
                b = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
                self.image = cv2.hconcat([a, b])

    def get_image(self) -> Optional[np.ndarray]:
        return self.image

    def save_image(self, file_path: str) -> None:
        if self.image is None:
            raise ValueError("No image data to save.")
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        cv2.imwrite(file_path, self.image)

    def extract_text(self, return_full_response: bool = False) -> Optional[str]:
        if self.image is None:
            return None
        client = self._ensure_client()

        success, encoded = cv2.imencode('.png', self.image)
        if not success:
            raise RuntimeError("Failed to encode image for OCR")
        content = encoded.tobytes()
        vision_image = vision.Image(content=content)
        response = client.text_detection(image=vision_image)

        if response.error.message:
            raise Exception(f"OCR API error: {response.error.message}")

        if return_full_response:
            return response

        texts = response.text_annotations
        return texts[0].description if texts else None

    def clear(self) -> None:
        self.image = None

    def export_image(self, file_path: str) -> None:
        if self.image is None:
            raise ValueError("No image data to export.")
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        cv2.imwrite(file_path, self.image)
