import cv2
import numpy as np
import time

YUNET_MODEL = "../models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "../models/face_recognition_sface_2021dec.onnx"
FACE_DB_PATH = "data/embeddings/me.npy"

CAMERA_ID = 0
INPUT_SIZE = (640, 480)

# 阈值先用 0.45~0.55 之间试
# 越高越严格，越低越容易误认
THRESHOLD = 0.50


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def mock_open_door(name, score):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OPEN DOOR: {name}, score={score:.3f}")


def main():
    known_embedding = np.load(FACE_DB_PATH)

    detector = cv2.FaceDetectorYN_create(
        YUNET_MODEL,
        "",
        INPUT_SIZE,
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )

    recognizer = cv2.FaceRecognizerSF_create(
        SFACE_MODEL,
        ""
    )

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE[1])

    last_open_time = 0
    open_interval = 3.0

    print("Camera opened.")
    print("Click the image window, then press q or ESC to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        h, w = frame.shape[:2]

        # YuNet 输入尺寸必须和当前 frame 尺寸一致
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)

                # SFace 推荐用 alignCrop，而不是手动裁剪
                aligned_face = recognizer.alignCrop(frame, face)
                current_embedding = recognizer.feature(aligned_face)
                current_embedding = current_embedding.flatten()

                score = cosine_similarity(current_embedding, known_embedding)

                if score >= THRESHOLD:
                    name = "me"
                    text = f"Authorized: {name} {score:.2f}"
                    color = (0, 255, 0)

                    now = time.time()
                    if now - last_open_time > open_interval:
                        mock_open_door(name, score)
                        last_open_time = now

                else:
                    text = f"Unknown {score:.2f}"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        cv2.imshow("face access demo", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()