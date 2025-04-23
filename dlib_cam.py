import cv2
import dlib
import numpy as np
import imutils
from picamera.array import PiRGBArray
from picamera import PiCamera
from scipy.spatial import distance as dist

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)  # Set resolution to 640x480 for better performance
camera.framerate = 32  # Adjust the framerate (lower for better performance)
raw_capture = PiRGBArray(camera, size=(640, 480))  # Raw capture buffer for PiCamera

# Load the face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper function to compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Helper function to compute the Mouth Aspect Ratio (MAR) for yawning
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Function to compute head pose angles (yaw, pitch, roll)
def get_head_pose_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# Function to compute head pose using 3D-2D correspondence
def get_head_pose(shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye left corner
        (225.0, 170.0, -135.0),    # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ])
    
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")

    camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # No lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector, translation_vector

# Start PiCamera stream and process frames
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    frame = frame.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])  # Convert to array of coordinates

        # Calculate EAR for drowsiness detection
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < 0.2:  # Threshold for detecting drowsiness
            cv2.putText(frame, "Drowsiness Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Calculate MAR for yawning detection
        mouth = shape[48:60]
        mar = mouth_aspect_ratio(mouth)
        if mar > 0.7:  # Threshold for detecting yawning
            cv2.putText(frame, "Yawning Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Calculate head pose (yaw, pitch, roll)
        rotation_vector, translation_vector = get_head_pose(shape)
        yaw, pitch, roll = get_head_pose_angles(rotation_vector)

        direction = ""
        if yaw < -15:
            direction = "Looking Left"
        elif yaw > 15:
            direction = "Looking Right"
        elif pitch < -15:
            direction = "Looking Down"
        elif pitch > 15:
            direction = "Looking Up"
        else:
            direction = "Facing Forward"

        cv2.putText(frame, f"Direction: {direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Draw face landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Show the frame with the annotations
    cv2.imshow("Frame", frame)

    # Clear the stream for the next frame
    raw_capture.truncate(0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
