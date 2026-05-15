import cv2

img = cv2.imread("data/test_images/test_face.jpg")

if img is None:
    print("read image failed")
else:
    print("image shape:", img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("gray shape:", gray.shape)
