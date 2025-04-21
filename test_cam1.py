import cv2

gst = (
    "libcamerasrc ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)

cap = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
print("hi")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    print("hi2")
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("hi3")
        break
print("hi4")
cap.release()
cv2.destroyAllWindows()
