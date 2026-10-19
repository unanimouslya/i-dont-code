import cv2
import mediapipe as mp
import numpy as np
from posture_detector import PostureDetector

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
CAMERA_INDEX = 0
WINDOW_TITLE = "Posture & Movement Module"

# Sleek visual styling palette (BGR)
COLOR_BONE = (245, 195, 60)         # Cyan / Sky-blue for shoulders and neck
COLOR_SPINE = (75, 230, 130)        # Lime green for spine column
COLOR_ARM = (100, 215, 255)         # Warm gold for arms
COLOR_JOINT = (255, 245, 140)       # Electric cyan joint centers
COLOR_JOINT_BORDER = (20, 25, 35)   # Dark contrast outline

# Head box colors
COLOR_BOX_NORMAL = (75, 225, 125)   # Green when head is within neutral zone
COLOR_BOX_DEVIATED = (60, 180, 255) # Amber when head has moved/tilted outside
COLOR_BOX_REF = (90, 95, 105)       # Dim gray for calibrated reference boundary

MIN_VISIBILITY = 0.45               # Visibility confidence required to render limbs


def draw_corner_brackets(frame, x1, y1, x2, y2, color, length=14, thickness=2):
    """Draws sleek corner brackets instead of a full closed box."""
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def draw_head_box(frame, data, w: int, h: int):
    """
    Renders the dynamic Head Movement Square and calibrated neutral zone.
    Visually isolates head movement within a bounding tracking square.
    """
    if not data.person_detected or data.current_head_pos is None:
        return

    # Box dimension in physical pixels
    box_px = int(data.head_box_size * w)
    half_box = box_px // 2

    # 1. Draw Calibrated Neutral Reference Box (where head was calibrated)
    if data.calibrated_head_pos:
        ref_cx, ref_cy = int(data.calibrated_head_pos[0] * w), int(data.calibrated_head_pos[1] * h)
        rx1, ry1 = ref_cx - half_box, ref_cy - half_box
        rx2, ry2 = rx1 + box_px, ry1 + box_px
        draw_corner_brackets(frame, rx1, ry1, rx2, ry2, COLOR_BOX_REF, length=12, thickness=1)

    # 2. Draw Current Real-Time Head Movement Square
    curr_cx, curr_cy = int(data.current_head_pos[0] * w), int(data.current_head_pos[1] * h)
    bx1, by1 = curr_cx - half_box, curr_cy - half_box
    bx2, by2 = bx1 + box_px, by1 + box_px

    box_color = COLOR_BOX_NORMAL if data.is_head_in_neutral_box else COLOR_BOX_DEVIATED

    # Draw tracking square outline
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)
    draw_corner_brackets(frame, bx1, by1, bx2, by2, (255, 255, 255), length=10, thickness=2)

    # Head center tracking crosshair/dot inside the square
    cv2.circle(frame, (curr_cx, curr_cy), 4, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
    cv2.circle(frame, (curr_cx, curr_cy), 2, box_color, -1, cv2.LINE_AA)

    # Label on top of the head square
    label = f"HEAD: {data.head_state}"
    cv2.putText(frame, label, (bx1, max(18, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)


def draw_skeleton(frame, landmarks, w: int, h: int):
    """
    Renders the clean upper-body skeleton overlay directly on the person.
    """
    def get_pt(idx):
        lm = landmarks.landmark[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        is_visible = (
            lm.visibility >= MIN_VISIBILITY
            and 0 <= px < w
            and 0 <= py < h
        )
        return (px, py), is_visible

    # Key upper-body landmarks
    pt_nose, v_nose = get_pt(0)
    pt_l_ear, v_l_ear = get_pt(7)
    pt_r_ear, v_r_ear = get_pt(8)
    pt_l_sh, v_l_sh = get_pt(11)
    pt_r_sh, v_r_sh = get_pt(12)

    if not (v_l_sh and v_r_sh):
        return

    # Suprasternal notch (Neck base / Mid-Shoulder)
    mid_sh = ((pt_l_sh[0] + pt_r_sh[0]) // 2, (pt_l_sh[1] + pt_r_sh[1]) // 2)
    sh_width = int(np.hypot(pt_r_sh[0] - pt_l_sh[0], pt_r_sh[1] - pt_l_sh[1]))

    # Head anchor (Nose or ear midpoint)
    if v_nose:
        head_anchor = pt_nose
    elif v_l_ear and v_r_ear:
        head_anchor = ((pt_l_ear[0] + pt_r_ear[0]) // 2, (pt_l_ear[1] + pt_r_ear[1]) // 2)
    else:
        head_anchor = (mid_sh[0], max(0, mid_sh[1] - int(sh_width * 0.55)))

    # 1. Shoulder Bar
    cv2.line(frame, pt_l_sh, pt_r_sh, COLOR_BONE, 3, cv2.LINE_AA)

    # 2. Neck Connection (Head to Suprasternal Notch)
    cv2.line(frame, head_anchor, mid_sh, COLOR_BONE, 3, cv2.LINE_AA)

    # 3. Thoracic Spine
    pt_l_hip, v_l_hip = get_pt(23)
    pt_r_hip, v_r_hip = get_pt(24)

    if v_l_hip and v_r_hip:
        mid_hip = ((pt_l_hip[0] + pt_r_hip[0]) // 2, (pt_l_hip[1] + pt_r_hip[1]) // 2)
        cv2.line(frame, mid_sh, mid_hip, COLOR_SPINE, 3, cv2.LINE_AA)
        cv2.line(frame, pt_l_sh, pt_l_hip, COLOR_BONE, 2, cv2.LINE_AA)
        cv2.line(frame, pt_r_sh, pt_r_hip, COLOR_BONE, 2, cv2.LINE_AA)
        cv2.line(frame, pt_l_hip, pt_r_hip, COLOR_BONE, 2, cv2.LINE_AA)

        for hip in [pt_l_hip, pt_r_hip]:
            cv2.circle(frame, hip, 6, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
            cv2.circle(frame, hip, 4, COLOR_JOINT, -1, cv2.LINE_AA)
    else:
        # Seated laptop view: proportional thoracic spine
        spine_len = int(sh_width * 0.65)
        torso_end = (mid_sh[0], min(h - 8, mid_sh[1] + spine_len))
        cv2.line(frame, mid_sh, torso_end, COLOR_SPINE, 3, cv2.LINE_AA)
        cv2.circle(frame, torso_end, 5, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
        cv2.circle(frame, torso_end, 3, (75, 230, 130), -1, cv2.LINE_AA)

    # 4. Arms (Shoulder -> Elbow -> Wrist)
    pt_l_el, v_l_el = get_pt(13)
    pt_l_wr, v_l_wr = get_pt(15)
    if v_l_el:
        cv2.line(frame, pt_l_sh, pt_l_el, COLOR_ARM, 2, cv2.LINE_AA)
        cv2.circle(frame, pt_l_el, 6, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
        cv2.circle(frame, pt_l_el, 4, COLOR_JOINT, -1, cv2.LINE_AA)
        if v_l_wr:
            cv2.line(frame, pt_l_el, pt_l_wr, COLOR_ARM, 2, cv2.LINE_AA)
            cv2.circle(frame, pt_l_wr, 5, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
            cv2.circle(frame, pt_l_wr, 3, COLOR_JOINT, -1, cv2.LINE_AA)

    pt_r_el, v_r_el = get_pt(14)
    pt_r_wr, v_r_wr = get_pt(16)
    if v_r_el:
        cv2.line(frame, pt_r_sh, pt_r_el, COLOR_ARM, 2, cv2.LINE_AA)
        cv2.circle(frame, pt_r_el, 6, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
        cv2.circle(frame, pt_r_el, 4, COLOR_JOINT, -1, cv2.LINE_AA)
        if v_r_wr:
            cv2.line(frame, pt_r_el, pt_r_wr, COLOR_ARM, 2, cv2.LINE_AA)
            cv2.circle(frame, pt_r_wr, 5, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
            cv2.circle(frame, pt_r_wr, 3, COLOR_JOINT, -1, cv2.LINE_AA)

    # 5. Core Joint Dots
    # Suprasternal notch joint
    cv2.circle(frame, mid_sh, 7, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
    cv2.circle(frame, mid_sh, 5, (75, 230, 130), -1, cv2.LINE_AA)

    # Shoulders
    for sh in [pt_l_sh, pt_r_sh]:
        cv2.circle(frame, sh, 8, COLOR_JOINT_BORDER, -1, cv2.LINE_AA)
        cv2.circle(frame, sh, 5, COLOR_JOINT, -1, cv2.LINE_AA)


def draw_hud(frame, data):
    """
    Renders the compact measurement panel with live frequency, speed, and activity sparkline.
    """
    h, w, _ = frame.shape

    panel_w = 255
    panel_h = min(370, h - 20)
    x1, y1 = 12, 12
    x2, y2 = x1 + panel_w, y1 + panel_h

    # Semi-transparent dark card
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 22, 28), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 60, 72), 1)

    c_white = (245, 245, 245)
    c_gray = (165, 170, 180)
    c_dim = (105, 110, 120)
    c_green = (75, 225, 125)
    c_amber = (60, 180, 255)
    c_blue = (235, 175, 75)
    c_red = (70, 70, 245)

    cv2.putText(frame, "BODY MOVEMENT", (x1 + 12, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, c_white, 2)
    cv2.line(frame, (x1 + 10, y1 + 32), (x2 - 10, y1 + 32), (50, 55, 65), 1)

    if not data.person_detected:
        cv2.putText(frame, "NO PERSON DETECTED", (x1 + 12, y1 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, c_red, 1)
        cv2.putText(frame, "Position yourself in camera view", (x1 + 12, y1 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, c_gray, 1)
        return

    # Section 1: Structural Alignment
    y = y1 + 52
    cv2.putText(frame, "HEAD:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    head_col = c_green if data.head_state == "CENTERED" else c_amber
    cv2.putText(frame, data.head_state, (x1 + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, head_col, 2)

    y += 20
    cv2.putText(frame, "SHOULDERS:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    sh_col = c_green if data.shoulder_state == "LEVEL" else c_amber
    cv2.putText(frame, data.shoulder_state, (x1 + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, sh_col, 2)

    y += 20
    cv2.putText(frame, "TORSO:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    torso_col = c_green if data.torso_state == "CENTERED" else c_amber
    cv2.putText(frame, data.torso_state, (x1 + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, torso_col, 2)

    # Divider
    y += 10
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 2: Current State & Dwell Time
    y += 18
    cv2.putText(frame, "CURRENT STATE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    state_col = c_green if data.current_state == "STABLE" else c_blue
    cv2.putText(frame, data.current_state, (x1 + 125, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, state_col, 2)

    y += 19
    cv2.putText(frame, "TIME IN STATE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    cv2.putText(frame, data.time_in_state_str, (x1 + 125, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, c_white, 2)

    # Divider
    y += 10
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 3: Movement Dynamics
    y += 18
    cv2.putText(frame, "MOVEMENT:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    mov_col = c_dim if data.movement_magnitude == "NONE" else (c_green if data.movement_magnitude == "SLIGHT" else c_amber)
    cv2.putText(frame, data.movement_magnitude, (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, mov_col, 2)

    y += 19
    cv2.putText(frame, "FREQUENCY:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    cv2.putText(frame, data.movement_frequency, (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, c_white, 2)

    y += 19
    cv2.putText(frame, "SPEED:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    cv2.putText(frame, data.movement_speed, (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, c_white, 1)

    # Divider
    y += 10
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 4: Repositioning Events
    y += 18
    cv2.putText(frame, "REPOSITIONING:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    cv2.putText(frame, str(data.repositioning_count), (x1 + 125, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, c_white, 2)

    y += 19
    cv2.putText(frame, "LAST CHANGE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_gray, 1)
    cv2.putText(frame, data.last_change_time, (x1 + 125, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, c_white, 1)

    # Divider
    y += 10
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 5: Activity Trend Sparkline
    y += 16
    cv2.putText(frame, "ACTIVITY TREND (45s):", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c_gray, 1)

    spark_x = x1 + 12
    spark_y = y + 24
    spark_w = panel_w - 24
    spark_h = 20

    # Sparkline base
    cv2.line(frame, (spark_x, spark_y), (spark_x + spark_w, spark_y), (45, 50, 60), 1)

    hist = data.activity_history
    if hist:
        bar_count = len(hist)
        bar_w = max(2, spark_w // bar_count)

        for i, val in enumerate(hist):
            bar_height = int(val * spark_h)
            bx = spark_x + (i * bar_w)
            by = spark_y - bar_height

            bar_col = (75, 180, 100) if val < 0.25 else ((60, 180, 255) if val < 0.65 else (60, 80, 255))
            cv2.line(frame, (bx, spark_y), (bx, by), bar_col, max(1, bar_w - 2))

    # Calibration hint
    y += 34
    hint_text = "[c] Recalibrate  |  [q] Exit"
    cv2.putText(frame, hint_text, (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 125, 135), 1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Unable to connect to webcam at index {CAMERA_INDEX}.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
        model_complexity=1,
        smooth_landmarks=True,
    )

    detector = PostureDetector()
    print("Posture & Movement Module running.")
    print("• Head Movement is visually enclosed in a tracking square.")
    print("• Press 'c' to instantly calibrate baseline to your current sitting posture.")
    print("• Press 'q' or 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        # Flip horizontally for natural mirror interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        pose_landmarks = None
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks

            # 1. Draw live clean 2D skeleton directly over person
            draw_skeleton(frame, pose_landmarks, w, h)

        # 2. Geometric & Movement Analysis
        posture_data = detector.update(pose_landmarks, w, h)

        # 3. Draw Head Movement Bounding Square (tracks head movement within a square)
        draw_head_box(frame, posture_data, w, h)

        # 4. Draw clean human-readable measurement panel & activity sparkline
        draw_hud(frame, posture_data)

        # 5. Display live window
        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            detector.calibrate()
            print("Recalibrated posture baseline to current position.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
