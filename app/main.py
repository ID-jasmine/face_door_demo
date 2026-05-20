import os
import sys
import time
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from src.config import load_config, get_config_value
from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase
from src.access_controller import AccessController,LedGPIO
from src.access_logger import AccessLogger
from src.status_store import StatusStore
from src.http_server import HttpServer


def abs_path(relative_path: str) -> str:
    """
    把 config.yaml 里的相对路径转换成项目根目录下的绝对路径。
    """
    return os.path.join(PROJECT_ROOT, relative_path)


class FrameBuffer:
    """
    线程安全的最新帧缓存。

    摄像头线程负责 set()
    识别线程负责 get()
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set(self, frame):
        if frame is None:
            return

        with self.lock:
            self.frame = frame.copy()

    def get(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()


def camera_capture_loop(
    camera,
    raw_frame_buffer,
    stop_event
):
    """
    摄像头采集线程。

    只负责：
    1. 持续读取摄像头
    2. 保存最新原始帧给识别线程

    注意：
    这里不做人脸检测。
    这里不做人脸识别。
    这里不画框。
    """
    print("[THREAD] Camera capture thread started.")

    while not stop_event.is_set():
        frame = camera.read()

        if frame is None:
            print("[WARN] Camera read returned None")
            time.sleep(0.05)
            continue

        # 保存最新原始帧，给识别线程使用
        raw_frame_buffer.set(frame)

        # 防止采集线程空转吃满 CPU
        time.sleep(0.005)

    print("[THREAD] Camera capture thread stopped.")


def recognition_loop(
    raw_frame_buffer,
    detector,
    recognizer,
    face_database,
    access_controller,
    access_logger,
    status_store,
    stop_event,
    recognition_interval=0.5
):
    """
    识别线程。

    每 recognition_interval 秒拿一张当前最新帧做：
    1. YuNet 人脸检测
    2. SFace 特征提取
    3. 本地人脸库比对
    4. 更新状态
    5. 模拟开锁 / 写日志

    注意：
    这里不更新网页视频画面。
    这里不画框。
    网页画面由 camera_capture_loop 推原始画面。
    """
    print("[THREAD] Recognition thread started.")

    last_recognition_time = 0
    last_face_present = False

    while not stop_event.is_set():
        now = time.time()

        if now - last_recognition_time < recognition_interval:
            time.sleep(0.02)
            continue

        last_recognition_time = now

        frame = raw_frame_buffer.get()

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            faces = detector.detect(frame)

            face_present = len(faces) > 0

            if not face_present:
                if last_face_present:
                    status_store.update_recognition(
                        "none",
                        0.0,
                        False
                    )

                access_controller.set_authorized(False)
                status_store.set_door_state("closed")

                if hasattr(status_store, "set_recognition_status"):
                    status_store.set_recognition_status("idle")

                last_face_present = False
                continue

            last_face_present = True

            # 当前阶段只处理第一张脸
            face = faces[0]

            embedding = recognizer.extract(frame, face)
            result = face_database.match(embedding)

            status_store.update_recognition(
                result.name,
                result.score,
                result.authorized
            )

            if result.authorized:
                changed = access_controller.set_authorized(True)
                status_store.set_door_state("opened")

                if changed:
                    access_logger.write(
                        result.name,
                        result.score,
                        authorized=True
                    )
            else:
                access_controller.set_authorized(False)
                status_store.set_door_state("closed")

                access_logger.write(
                    result.name,
                    result.score,
                    authorized=False
                )

            # 不要每次都 print，避免 SSH 终端刷屏导致卡顿
            # print(
            #     f"name={result.name}, "
            #     f"score={result.score:.3f}, "
            #     f"authorized={result.authorized}"
            # )

        except Exception as e:
            print("[ERROR] recognition_loop failed:", e)

            if hasattr(status_store, "set_recognition_status"):
                status_store.set_recognition_status("error")

            time.sleep(0.2)

    print("[THREAD] Recognition thread stopped.")


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

    # 识别频率：默认每 0.5 秒识别一次
    recognition_interval = get_config_value(
        config,
        "recognition.interval",
        0.5
    )

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
        gpio=LedGPIO(led_name="status_led", active_high=True),
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

    try:
        camera.open()
    except RuntimeError as e:
        print("[ERROR] camera.open() failed:", e)
        print("[ERROR] Cannot start camera thread.")
        return

    print("Config loaded.")
    print(f"Camera: id={camera_id}, width={camera_width}, height={camera_height}")
    print(f"YuNet model: {yunet_model}")
    print(f"SFace model: {sface_model}")
    print(f"Face DB: {face_db_path}")
    print(f"Recognition interval: {recognition_interval}s")
    print("Camera opened.")
    print("Face detector loaded.")
    print("HTTP server started.")
    print("Press Ctrl+C to quit.")

    # =========================
    # 4. 启动线程
    # =========================

    raw_frame_buffer = FrameBuffer()
    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=camera_capture_loop,
        args=(
            camera,
            raw_frame_buffer,
            stop_event
        ),
        daemon=True
    )

    recognition_thread = threading.Thread(
        target=recognition_loop,
        args=(
            raw_frame_buffer,
            detector,
            recognizer,
            face_database,
            access_controller,
            access_logger,
            status_store,
            stop_event
        ),
        kwargs={
            "recognition_interval": recognition_interval
        },
        daemon=True
    )

    camera_thread.start()
    recognition_thread.start()

    # =========================
    # 5. 主线程保持运行
    # =========================

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MAIN] Stopping...")
        stop_event.set()
        time.sleep(0.5)
    finally:
        try:
            access_controller.set_authorized(False)
        except Exception as e:
            print("[WARN] failed to turn off LED:", e)

        camera.release()
        print("[MAIN] Camera released.")
        print("[MAIN] Exit.")


if __name__ == "__main__":
    main()