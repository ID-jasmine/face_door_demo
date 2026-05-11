import cv2
import os
import time

SAVE_DIR = "data/faces/me"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Cannot load face cascade")
        return

    count = 0

    print("Camera opened.")
    print("Press s to save face, q or ESC to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=8,
            minSize=(100, 100)
        )

        # 如果检测到多张脸，只取最大的那张
        if len(faces) > 0:
            faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
            x, y, w, h = faces[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "press s to save",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("collect faces", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if len(faces) == 0:
                print("No face detected, skip.")
                continue

            x, y, w, h = faces[0]

            face_img = frame[y:y + h, x:x + w]

            filename = f"face_{int(time.time())}_{count}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)

            cv2.imwrite(save_path, face_img)
            count += 1

            print(f"Saved: {save_path}")

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()