import cv2


class FaceDetector:
    def __init__(
        self,
        model_path,
        input_size=(640, 480),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    ):
        self.model_path = model_path
        self.input_size = input_size

        self.detector = cv2.FaceDetectorYN_create(
            self.model_path,
            "",
            self.input_size,
            score_threshold,
            nms_threshold,
            top_k
        )

    def detect(self, frame):
        if frame is None:
            return []

        h, w = frame.shape[:2]

        # YuNet 要求输入尺寸和当前 frame 尺寸匹配
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        return faces

    @staticmethod
    def draw_faces(frame, faces, color=(0, 255, 0)):
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                "face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return frame