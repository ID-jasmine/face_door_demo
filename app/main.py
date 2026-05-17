import os
import sys

import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.config import load_config, get_config_value

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase
from src.access_controller import AccessController
from src.access_logger import AccessLogger
from src.status_store import StatusStore
from src.http_server import HttpServer


def abs_path(relative_path: str) -> str:
    """
    把 config.yaml 里的相对路径转换成项目根目录下的绝对路径。
    """
    return os.path.join(PROJECT_ROOT, relative_path)


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))

    # =========================
    # 1. 从配置文件读取参数
    # =========================

    camera_id = get_config_value(config, "camera.device_index", 0)
    camera_width = get_config_value(config, "camera.width", 640)
    camera_height = get_config_value(config, "camera.height", 480)

    yunet_model = abs_path(
        get_config_value(
            config,
            "model.yunet",
            "models/face_detection_yunet_2023mar.onnx"
        )
    )

    sface_model = abs_path(
        get_config_value(
            config,
            "model.sface",
            "models/face_recognition_sface_2021dec.onnx"
        )
    )

    face_db_path = abs_path(
        get_config_value(
            config,
            "face_database.embedding_path",
            "data/embeddings/me.npy"
        )
    )

    face_name = get_config_value(config, "face_database.name", "me")
    face_threshold = get_config_value(config, "face_database.threshold", 0.50)

    detector_score_threshold = get_config_value(
        config,
        "face_detector.score_threshold",
        0.8
    )

    open_duration = get_config_value(config, "access.open_duration", 0.5)
    cooldown = get_config_value(config, "access.cooldown", 3.0)

    access_log_path = abs_path(
        get_config_value(
            config,
            "log.access_log_path",
            "data/logs/access.log"
        )
    )

    unknown_cooldown = get_config_value(config, "log.unknown_cooldown", 3.0)

    http_host = get_config_value(config, "http_server.host", "0.0.0.0")
    http_port = get_config_value(config, "http_server.port", 8000)

    # =========================
    # 2. 创建各个模块对象
    # =========================

    camera = Camera(
        camera_id=camera_id,
        width=camera_width,
        height=camera_height
    )

    recognizer = FaceRecognizer(
        model_path=sface_model
    )

    detector = FaceDetector(
        model_path=yunet_model,
        input_size=(camera_width, camera_height),
        score_threshold=detector_score_threshold
    )

    face_database = FaceDatabase(
        embedding_path=face_db_path,
        name=face_name,
        threshold=face_threshold
    )

    access_controller = AccessController(
        open_duration=open_duration,
        cooldown=cooldown
    )

    access_logger = AccessLogger(
        access_log_path,
        unknown_cooldown=unknown_cooldown
    )

    state_file = abs_path(
        get_config_value(
            config,
            "state.file",
            "runtime/state.json"
        )
    )

    status_store = StatusStore(
        state_file=state_file
    )

    http_server = HttpServer(
        status_store=status_store,
        host=http_host,
        port=http_port
    )
    http_server.start()

    # =========================
    # 3. 打开摄像头
    # =========================

    camera_ok = True

    try:
        camera.open()
    except RuntimeError as e:
        camera_ok = False
        print("[WARN] camera.open() failed:", e)
        print("[WARN] Web server will start without camera.")
        # print("[ERROR]", e)
        # return

    print("Config loaded.")
    print(f"Camera: id={camera_id}, width={camera_width}, height={camera_height}")
    print(f"YuNet model: {yunet_model}")
    print(f"SFace model: {sface_model}")
    print(f"Face DB: {face_db_path}")
    print("Camera opened.")
    print("Face detector loaded.")
    print("Click the image window, then press q or ESC to quit.")

    # =========================
    # 4. 主循环
    # =========================

    while True:
        if not camera_ok:
            print("[WARN] Camera not available")
            time.sleep(1)
            continue
        frame = camera.read()

        if frame is None:
            print("[ERROR] Cannot read frame")
            camera_ok = False
            continue

        faces = detector.detect(frame)
        frame = detector.draw_faces(frame, faces)

        http_server.update_latest_frame(frame)
        
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

            # 当前阶段只处理画面中检测到的第一张脸
            break

        #cv2.imshow("camera module test", frame)

        # key = cv2.waitKey(1) & 0xFF

        # if key == ord("q") or key == 27:
        #     break

    camera.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()