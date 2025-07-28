import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def process_deadlift_tracker():
    cap = cv2.VideoCapture(0)
    rep_count = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angles
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Feedback variables
                feedback = ""
                hip_color = (0, 255, 0)  # Default green for correct position
                knee_color = (0, 255, 0)
                elbow_color = (0, 255, 0)  # Set default to green for a straight elbow

                # Logic for determining good/bad form
                if hip_angle > 160 and stage == "down":
                    stage = "up"
                    rep_count += 1
                    feedback = "Great! You've completed a rep!"
                elif hip_angle < 90:
                    stage = "down"

                # Check for good form and update colors if needed
                if hip_angle > 160 and knee_angle > 160:
                    feedback = "Great! You've reached the lockout position."
                else:
                    if hip_angle <= 90 or hip_angle > 160:
                        hip_color = (0, 0, 255)  # Red for incorrect hip position
                        feedback = "Adjust your hip position!"
                    if knee_angle <= 120 or knee_angle > 160:
                        knee_color = (0, 0, 255)  # Red for incorrect knee position
                        feedback = "Adjust your knee position!"

                # Check for correct elbow form - red only if bent too much
                if elbow_angle < 160:  # If elbow is bent more than desired
                    elbow_color = (0, 0, 255)  # Red for bent elbow
                    feedback += " Keep your elbow straight!"

                # Draw landmarks and connections selectively in red only for incorrect positions
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Default green
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
                )

                # Highlight incorrect landmarks in red
                if hip_color == (0, 0, 255):  # If hip is incorrect
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks,
                        [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE)],
                        mp_drawing.DrawingSpec(color=hip_color, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=hip_color, thickness=2, circle_radius=4)
                    )

                if knee_color == (0, 0, 255):  # If knee is incorrect
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks,
                        [(mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)],
                        mp_drawing.DrawingSpec(color=knee_color, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=knee_color, thickness=2, circle_radius=4)
                    )

                if elbow_color == (0, 0, 255):  # If elbow is incorrect
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks,
                        [(mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)],
                        mp_drawing.DrawingSpec(color=elbow_color, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=elbow_color, thickness=2, circle_radius=4)
                    )

                # Display feedback and rep count
                cv2.putText(image, feedback, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Reps: {rep_count}', (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(e)

            # Show the image
            cv2.imshow('Deadlift Tracker', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the deadlift tracker
process_deadlift_tracker()