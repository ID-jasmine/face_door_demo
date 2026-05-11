import os
import cv2
import numpy as np

FACE_DIR = "data/faces/me"
OUTPUT_PATH = "data/embeddings/me.npy"

SFACE_MODEL = "../models/face_recognition_sface_2021dec.onnx"


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def main():
    if not os.path.exists(FACE_DIR):
        print(f"Face dir not found: {FACE_DIR}")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    recognizer = cv2.FaceRecognizerSF_create(
        SFACE_MODEL,
        ""
    )

    embeddings = []

    image_names = sorted(os.listdir(FACE_DIR))

    for name in image_names:
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(FACE_DIR, name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        # 这里你的图片已经是裁剪出来的人脸，所以先直接送给 SFace
        try:
            face_feature = recognizer.feature(img)
        except Exception as e:
            print(f"Feature extract failed: {img_path}, error: {e}")
            continue

        face_feature = face_feature.flatten()
        face_feature = l2_normalize(face_feature)

        embeddings.append(face_feature)
        print(f"Loaded: {img_path}, feature shape: {face_feature.shape}")

    if len(embeddings) == 0:
        print("No valid face embeddings generated.")
        return

    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = l2_normalize(mean_embedding)

    np.save(OUTPUT_PATH, mean_embedding)

    print()
    print(f"Saved face database: {OUTPUT_PATH}")
    print(f"Used images: {len(embeddings)}")
    print(f"Embedding shape: {mean_embedding.shape}")


if __name__ == "__main__":
    main()
    