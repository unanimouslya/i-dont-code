import cv2
import mediapipe as mp
from blink_detector import (
    BlinkDetector,
    calculate_ear,
    get_landmark_points,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    DEFAULT_EAR_THRESHOLD,
)

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
CAMERA_INDEX = 0                       # Standard default webcam
EAR_THRESHOLD = DEFAULT_EAR_THRESHOLD  # Eye Aspect Ratio threshold (OPEN vs CLOSED)


def main():
    # 1. Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Unable to open webcam at index {CAMERA_INDEX}.")
        return

    # 2. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize the state-machine blink detector
    detector = BlinkDetector(ear_threshold=EAR_THRESHOLD)

    print("Blink detection running. Press 'q' or 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        # Flip horizontally for a natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # 12. Handle face detection / no-face frames
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # 3. Identify eye landmark coordinates
            left_points = get_landmark_points(face_landmarks, LEFT_EYE_LANDMARKS, w, h)
            right_points = get_landmark_points(face_landmarks, RIGHT_EYE_LANDMARKS, w, h)

            # 10. Draw eye landmarks on webcam image
            for point in left_points + right_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # 4. Calculate EAR for each eye
            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)

            # 5. Calculate average EAR
            avg_ear = (left_ear + right_ear) / 2.0

            # 6, 7, 8. Update open/closed state & detect blinks
            detector.update(avg_ear)

            state_label = f"State: {detector.state}"
            state_color = (0, 255, 0) if detector.state == "OPEN" else (0, 0, 255)
            left_ear_text = f"Left EAR: {left_ear:.2f}"
            right_ear_text = f"Right EAR: {right_ear:.2f}"
            avg_ear_text = f"Avg EAR: {avg_ear:.2f}"
        else:
            # No face detected in current frame
            state_label = "State: NO FACE"
            state_color = (0, 165, 255)
            left_ear_text = "Left EAR: N/A"
            right_ear_text = "Right EAR: N/A"
            avg_ear_text = "Avg EAR: N/A"

        # 11. Display metrics on webcam feed
        cv2.putText(frame, f"Blinks: {detector.blink_count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, state_label, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(frame, avg_ear_text, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, left_ear_text, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, right_ear_text, (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 9. Display the webcam feed
        cv2.imshow("Blink Detector", frame)

        # Exit loop on 'q' or ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
