
import cv2
import mediapipe as mp
import random

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define acupuncture points with more accurate indices
acupuncture_points = {
    "Stress Relief": 168,      # Between eyebrows
    "Calm Mind": 17,           # Below lower lip
    "Soothing Eyes": 10,       # Around the eyes
    "Smile Point": 61,         # Corner of mouth
    "Temple Massage": 46,      # Temple region
    "Jaw Relaxer": 234,        # Jawline
    "Forehead Relaxation": 151, # Mid-forehead
    "Cheek Comfort": 94,       # Cheek area
    "Chin Point": 17,          # Below lower lip
}
# Assign a unique color to each acupuncture point
point_colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in acupuncture_points}

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Variables to store feedback
feedback_text = ""

# Initialize MediaPipe Face Mesh and Hands
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,  # Limiting to 1 face for better performance
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as face_mesh, \
        mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,  # Detect only one hand
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_resized = cv2.resize(frame, (1280, 720))
        frame_resized = cv2.flip(frame_resized, 1)
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process the frame with Face Mesh and Hands
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        h, w, _ = frame_resized.shape

        # Draw only acupuncture points on the face
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw specific acupuncture points with labels
                for point_name, index in acupuncture_points.items():
                    landmark = face_landmarks.landmark[index]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    color = point_colors[point_name]
                    cv2.circle(frame_resized, (x, y), 8, color, -1)
                    cv2.putText(frame_resized, point_name, (x + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Calculate dynamic tolerance based on face size
                face_width = abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x) * w
                tolerance = int(face_width * 0.05)  # 5% of face width as tolerance

        # Detect fingertip for acupressure interaction
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Index fingertip (Landmark 8)
                fingertip = hand_landmarks.landmark[8]
                fx, fy = int(fingertip.x * w), int(fingertip.y * h)
                cv2.circle(frame_resized, (fx, fy), 10, (255, 0, 0), -1)

                # Check if fingertip is near any acupressure point
                if face_results.multi_face_landmarks:
                    for point_name, index in acupuncture_points.items():
                        landmark = face_landmarks.landmark[index]
                        px, py = int(landmark.x * w), int(landmark.y * h)

                        # Calculate distance between fingertip and acupressure point
                        distance = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5
                        if distance <= tolerance:
                            feedback_text = f"Good! Pressing {point_name}"
                            break
                    else:
                        feedback_text = "Not on the correct point!"

        # Display the feedback text on the frame
        cv2.putText(frame_resized, feedback_text, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Acupressure Points Detection', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()