import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer

YUNET_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_recognition_sface_2021dec.onnx"
)

def main():
    camera = Camera(camera_id=0, width=640, height=480)

    recognizer = FaceRecognizer(
        model_path=SFACE_MODEL
    )

    detector = FaceDetector(
        model_path=YUNET_MODEL,
        input_size=(640, 480),
        score_threshold=0.8
    )

    try:
        camera.open()
    except RuntimeError as e:
        print("[ERROR]", e)
        return

    print("Camera opened.")
    print("Face detector loaded.")
    print("Click the image window, then press q or ESC to quit.")
    
    while True:
        frame = camera.read()

        if frame is None:
            print("[ERROR] Cannot read frame")
            break
        
        faces = detector.detect(frame)
        frame = detector.draw_faces(frame, faces)

        for face in faces:
            embedding = recognizer.extract(frame, face)
            print("Embedding shape:", embedding.shape)
            break
        
        cv2.imshow("camera module test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()