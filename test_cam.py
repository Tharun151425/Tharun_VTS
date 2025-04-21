import cv2
cap = cv2.VideoCapture(0)  # Open the default camera (0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)
    else:
        print("Error: Could not grab frame.")
cap.release()
cv2.destroyAllWindows()
