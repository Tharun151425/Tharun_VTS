import time
import dlib
import cv2
import threading
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

# Initialize the camera and dlib detector
camera = PiCamera()
camera.resolution = (640, 480)  # Set resolution to 640x480 for performance
camera.framerate = 15  # Set FPS to 15 (you can adjust based on performance)
raw_capture = PiRGBArray(camera, size=(640, 480))

# Initialize dlib detector and shape predictor for facial landmarks
detector = dlib.get_frontal_face_detector()  # HOG-based detector for efficiency
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio thresholds for drowsiness detection
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_CONSECUTIVE_FRAMES = 10
MOUTH_THRESHOLD = 0.5

frame_counter = 0
drowsy_counter = 0
yawn_counter = 0

def eye_aspect_ratio(eye):
    # Calculate the Euclidean distance between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the mouth aspect ratio (MAR) based on the distance between mouth landmarks
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    return (A + B) / 2.0

def process_frame(frame):
    global drowsy_counter, yawn_counter

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    for face in faces:
        # Get face landmarks
        landmarks = predictor(gray, face)

        # Get the eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Calculate the EAR (Eye Aspect Ratio)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Drowsiness Detection
        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            drowsy_counter += 1
            if drowsy_counter >= EYE_CONSECUTIVE_FRAMES:
                cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            drowsy_counter = 0

        # Get the mouth landmarks for yawning detection
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])
        mar = mouth_aspect_ratio(mouth)

        # Yawning Detection
        if mar > MOUTH_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= EYE_CONSECUTIVE_FRAMES:
                cv2.putText(frame, "Yawning Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            yawn_counter = 0

        # Head Pose Estimation
        # Use the nose tip as the reference point for the head pose
        nose_end_point = (0, 0, 1000.0)
        camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]])  # Example camera matrix
        dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion for simplicity

        # 3D points for pose estimation (nose tip)
        nose_end_point_2D, _ = cv2.projectPoints(np.array([nose_end_point]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Determine orientation based on rotation vector and angles
        angles = cv2.Rodrigues(rotation_vector)[0]
        pitch, yaw, roll = angles[0, 0], angles[1, 0], angles[2, 0]
        
        # Show the head direction (left, right, up, down)
        if pitch < -10:
            cv2.putText(frame, "Looking Down", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif pitch > 10:
            cv2.putText(frame, "Looking Up", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif yaw < -10:
            cv2.putText(frame, "Looking Left", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif yaw > 10:
            cv2.putText(frame, "Looking Right", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Facing Forward", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw the landmarks and face rectangles for visualization
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        for i in range(36, 48):  # Draw eye landmarks
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (255, 0, 0), -1)
        for i in range(48, 60):  # Draw mouth landmarks
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (255, 0, 0), -1)

    # Display the processed frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

def capture_frames():
    global frame_counter
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        frame_counter += 1
        if frame_counter % 2 == 0:  # Skip every 2nd frame to reduce load
            continue
        
        frame = frame.array
        process_frame(frame)  # Process the frame for drowsiness, yawning, head pose, etc.
        
        # Clear the stream for the next frame
        raw_capture.truncate(0)

# Start capture in a separate thread to handle real-time frame processing
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Keep the script running and process frames continuously
while True:
    time.sleep(1)  # The main thread sleeps, but the capture and processing continue in the background
