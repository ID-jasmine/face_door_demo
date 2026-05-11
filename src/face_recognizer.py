import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, model_path):
        self.model_path = model_path

        self.recognizer = cv2.FaceRecognizerSF_create(
            self.model_path,
            ""
        )

    def extract(self, frame, face):
        """
        frame: 原始摄像头画面
        face: YuNet 检测出来的一张脸，包含 bbox 和关键点
        return: 归一化后的 embedding
        """
        aligned_face = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned_face)

        feature = feature.flatten()
        feature = self._l2_normalize(feature)

        return feature

    @staticmethod
    def _l2_normalize(vec):
        norm = np.linalg.norm(vec)

        if norm == 0:
            return vec

        return vec / norm