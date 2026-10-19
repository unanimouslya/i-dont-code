import math

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
# Eye Aspect Ratio threshold below which an eye is considered CLOSED.
# Values typically range between 0.18 and 0.23 depending on individual eye shape.
DEFAULT_EAR_THRESHOLD = 0.21

# MediaPipe Face Mesh 6-point indices for each eye:
# Format: [corner_1, top_1, top_2, corner_2, bottom_2, bottom_1]
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]


def get_landmark_points(face_landmarks, indices, frame_width, frame_height):
    """
    Extracts 2D pixel coordinates (x, y) for a specified list of landmark indices.

    :param face_landmarks: MediaPipe normalized face landmarks object.
    :param indices: List of integer landmark indices to extract.
    :param frame_width: Width of the camera image in pixels.
    :param frame_height: Height of the camera image in pixels.
    :return: List of (x, y) integer tuples.
    """
    points = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        points.append((x, y))
    return points


def calculate_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye using 6 landmark points.

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    :param eye_points: List of 6 (x, y) coordinates:
                       [p1 (corner), p2 (upper), p3 (upper),
                        p4 (corner), p5 (lower), p6 (lower)]
    :return: Float value representing EAR.
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    # Compute Euclidean distances between vertical eyelid points
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    # Compute Euclidean distance between horizontal eye corner points
    horizontal = math.dist(p1, p4)

    # Prevent division by zero if eye corner points overlap
    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


class BlinkDetector:
    """
    Manages blink detection using an OPEN -> CLOSED -> OPEN state transition.
    """

    def __init__(self, ear_threshold=DEFAULT_EAR_THRESHOLD):
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.state = "OPEN"  # Tracks current state: "OPEN" or "CLOSED"

    def update(self, avg_ear):
        """
        Updates the state machine based on the current frame's average EAR.
        A blink is counted only when the eye completes: OPEN -> CLOSED -> OPEN.

        :param avg_ear: Average EAR across both eyes.
        :return: Boolean indicating whether a blink was completed on this frame.
        """
        blink_detected = False

        if avg_ear < self.ear_threshold:
            # Eyes transitioned to or remain in CLOSED state
            self.state = "CLOSED"
        else:
            # Eyes are currently OPEN
            if self.state == "CLOSED":
                # Completed the full cycle: OPEN -> CLOSED -> OPEN
                self.blink_count += 1
                blink_detected = True
            self.state = "OPEN"

        return blink_detected
