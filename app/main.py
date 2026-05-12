import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase
from src.access_controller import AccessController
from src.access_logger import AccessLogger
from src.status_store import StatusStore
from src.http_server import HttpServer

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

FACE_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "embeddings",
    "me.npy"
)

def main():
    camera = Camera(
        camera_id=0, 
        width=640, 
        height=480
    )

    recognizer = FaceRecognizer(
        model_path=SFACE_MODEL
    )

    detector = FaceDetector(
        model_path=YUNET_MODEL,
        input_size=(640, 480),
        score_threshold=0.8
    )

    face_database = FaceDatabase(
        embedding_path=FACE_DB_PATH,
        name="me",
        threshold=0.50
    )

    access_controller = AccessController(
        open_duration=0.5,
        cooldown=3.0
    )

    access_logger = AccessLogger(
        os.path.join(PROJECT_ROOT, "data", "logs", "access.log"),
        unknown_cooldown=3.0
    )

    status_store = StatusStore()

    http_server = HttpServer(
        status_store=status_store,
        host="0.0.0.0",
        port=8000
    )
    http_server.start()

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
            result = face_database.match(embedding)
            status_store.update_recognition(
                result.name,
                result.score,
                result.authorized
            )

            if result.authorized:
                opened = access_controller.open_door(
                    result.name,
                    result.score
                )

                if opened:
                    status_store.set_door_state("opened")
                    access_logger.write(
                        result.name,
                        result.score,
                        authorized=True
                    )
                    status_store.set_door_state("closed")
            else:
                access_logger.write(
                    result.name,
                    result.score,
                    authorized=False
                )

            print(
                f"name={result.name}, "
                f"score={result.score:.3f}, "
                f"authorized={result.authorized}"
            )
            break
        
        cv2.imshow("camera module test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()