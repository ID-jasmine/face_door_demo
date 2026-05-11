import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open /dev/video0")
        return

    print("Camera opened. Press q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        cv2.imshow("camera preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
