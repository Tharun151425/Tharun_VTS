from picamera2 import Picamera2
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

time.sleep(1)  # Give camera time to warm up

# Load dlib stuff
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR function
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Head pose functions
def get_head_pose(shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye
        (225.0, 170.0, -135.0),    # Right eye
        (-150.0, -150.0, -125.0),  # Left mouth
        (150.0, -150.0, -125.0)    # Right mouth
    ])
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")
    camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector

def get_head_pose_angles(rotation_vector):
    rot_mat, _ = cv2.Rodrigues(rotation_vector)
    pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
    yaw = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2))
    roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# Main loop
while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # EAR check
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        if ear < 0.2:
            cv2.putText(frame, "Drowsiness Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # MAR check
        mouth = shape[48:60]
        mar = mouth_aspect_ratio(mouth)
        if mar > 0.7:
            cv2.putText(frame, "Yawning Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Head pose
        rotation_vector = get_head_pose(shape)
        yaw, pitch, roll = get_head_pose_angles(rotation_vector)
        direction = "Facing Forward"
        if yaw < -15: direction = "Looking Left"
        elif yaw > 15: direction = "Looking Right"
        elif pitch < -15: direction = "Looking Down"
        elif pitch > 15: direction = "Looking Up"
        cv2.putText(frame, f"Direction: {direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Dlib Cam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

