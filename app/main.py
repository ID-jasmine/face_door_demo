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
from src.access_controller import AccessController
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
    http_server,
    stop_event,
    video_interval=0.1
):
    """
    摄像头采集线程。

    只负责：
    1. 持续读取摄像头
    2. 保存最新原始帧给识别线程
    3. 定期把原始画面推给网页

    注意：
    这里不做人脸检测。
    这里不做人脸识别。
    这里不画框。
    """
    print("[THREAD] Camera capture thread started.")

    last_video_update_time = 0

    while not stop_event.is_set():
        frame = camera.read()

        if frame is None:
            print("[WARN] Camera read returned None")
            time.sleep(0.05)
            continue

        # 保存最新原始帧，给识别线程使用
        raw_frame_buffer.set(frame)

        # 网页视频显示原始画面，不画框
        now = time.time()
        if now - last_video_update_time >= video_interval:
            last_video_update_time = now
            http_server.update_latest_frame(frame)

        # 防止采集线程空转吃满 CPU
        time.sleep(0.005)

    print("[THREAD] Camera capture thread stopped.")


def open_door_worker(access_controller, access_logger, status_store, name, score):
    """
    开门线程。

    避免 open_door 里面的 sleep 阻塞识别线程。
    """
    opened = access_controller.open_door(name, score)

    if opened:
        status_store.set_door_state("opened")

        access_logger.write(
            name,
            score,
            authorized=True
        )

        status_store.set_door_state("closed")


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

            if len(faces) == 0:
                if hasattr(status_store, "set_recognition_status"):
                    status_store.set_recognition_status("idle")
                continue

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
                threading.Thread(
                    target=open_door_worker,
                    args=(
                        access_controller,
                        access_logger,
                        status_store,
                        result.name,
                        result.score
                    ),
                    daemon=True
                ).start()
            else:
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

    # 网页视频刷新频率：默认 0.1 秒更新一次原始画面，大约 10 FPS
    video_interval = get_config_value(
        config,
        "video.interval",
        0.1
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
    print(f"Video update interval: {video_interval}s")
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
            http_server,
            stop_event
        ),
        kwargs={
            "video_interval": video_interval
        },
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
        camera.release()
        print("[MAIN] Camera released.")
        print("[MAIN] Exit.")


if __name__ == "__main__":
    main()