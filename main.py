"""
4x3 Calibrated Eye Tracking Module - Main Application
Focus Detection Model for Students - Behavioral Sensing Module

Features:
- High-Resolution Eye Crop: Focuses on the eye orbit at 1280x720 HD resolution
- Sub-Pixel Ocular Displacement within the bony orbital socket
- Ocular-Morphology-Invariant: Completely immune to monolids, epicanthic folds, and hooded lids
- 11.7 mm Corneal Limbus Invariant: Scales camera pixels into real physical millimeters of eye rotation
- Direct Biomechanical Transfer Function (Zero L2 Shrinkage Bias)
- Live Magnified Zoomed-In Eye Monitor in Picture-in-Picture [v to toggle]
- Fullscreen Mode support [f to toggle] for maximum visual angle and SNR
"""

import math
import sys
import time
import cv2
import mediapipe as mp
import numpy as np

from gaze_calibrator import (
    GazeGridCalibrator,
    CalibratedGazeData,
    GRID_COLS,
    GRID_ROWS,
    ZONE_LABELS,
    HUMAN_IRIS_DIAMETER_MM,
)


# ==============================================================================
# CONFIGURABLE PARAMETERS & VISUAL THEME
# ==============================================================================
CAMERA_INDEX = 0
WINDOW_TITLE = "4x3 Calibrated Eye Tracking"
ALWAYS_CALIBRATE_ON_START = True       # Runs 2-phase calibration on every startup

# Default Display Canvas Dimensions (fits 1280x800 screens comfortably)
DEFAULT_CANVAS_W = 1260
DEFAULT_CANVAS_H = 720

# Color Palette (BGR)
COLOR_BG = (18, 20, 26)                # Dark slate workspace
COLOR_GRID_LINE = (38, 42, 54)         # Cell grid divider lines
COLOR_CELL_BG = (22, 25, 33)           # Cell default background
COLOR_CELL_ACTIVE_BG = (45, 100, 60)   # Highlight wash for currently focused cell
COLOR_CELL_ACTIVE_BORDER = (75, 225, 125) # Glowing active cell border (Green/Cyan)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (165, 170, 180)
COLOR_DIM = (95, 100, 115)
COLOR_TEXT_MAIN = COLOR_WHITE
COLOR_TEXT_MUTED = COLOR_GRAY
COLOR_TEXT_DIM = COLOR_DIM
COLOR_CYAN = (245, 195, 60)            # Cyan/Gold accent
COLOR_GREEN = (75, 225, 125)           # Active / stable green
COLOR_AMBER = (60, 180, 255)           # Pauses / warnings
COLOR_RED = (60, 60, 245)              # Alerts / errors

# Target Pulsing Animation parameters
PULSE_SPEED = 4.0


def draw_hud(frame, data: CalibratedGazeData, hud_x: int = 14, hud_y: int = 14):
    """
    Renders the compact project HUD card displaying calibrated gaze metrics,
    cognitive attention states, pupillometry, blink dynamics, and 3D head pose.
    """
    panel_w = 300
    panel_h = 445
    x1, y1 = hud_x, hud_y
    x2, y2 = x1 + panel_w, y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 22, 28), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 60, 72), 1)

    # Header
    y = y1 + 22
    cv2.putText(frame, "ZOOMED EYE GAZE (4x3)", (x1 + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_WHITE, 2, cv2.LINE_AA)

    status_col = COLOR_GREEN if (data.is_calibrated and data.face_detected) else COLOR_AMBER
    cv2.circle(frame, (x2 - 16, y - 5), 4, status_col, -1, cv2.LINE_AA)

    y += 8
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    if not data.face_detected:
        cv2.putText(frame, "NO FACE DETECTED", (x1 + 12, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_RED, 1, cv2.LINE_AA)
        cv2.putText(frame, "Position face in camera view", (x1 + 12, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        return

    # Section 1: Active Zone & Label
    y += 18
    cv2.putText(frame, "ACTIVE ZONE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    zone_str = f"Row {data.active_grid[0]}, Col {data.active_grid[1]}"
    cv2.putText(frame, zone_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GREEN, 2, cv2.LINE_AA)

    y += 18
    cv2.putText(frame, "REGION:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    short_label = data.active_label.split("(")[0].strip()
    cv2.putText(frame, short_label, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)

    y += 18
    cv2.putText(frame, "CONFIDENCE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    conf_str = f"{data.confidence_pct}% (RANSAC: {data.ransac_inlier_pct:.0f}%)"
    cv2.putText(frame, conf_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_WHITE, 1, cv2.LINE_AA)

    y += 8
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 2: Cognitive & Behavioral Attention State
    y += 17
    cv2.putText(frame, "GAZE STATE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    if data.gaze_state == "FIXATION":
        gaze_col = COLOR_GREEN
        gaze_lbl = "FIXATION (STEADY)"
    elif data.gaze_state == "DARTING":
        gaze_col = COLOR_AMBER
        gaze_lbl = "DARTING (RESTLESS)"
    else:
        gaze_col = COLOR_CYAN
        gaze_lbl = "SACCADE (MOVING)"
    cv2.putText(frame, gaze_lbl, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gaze_col, 2, cv2.LINE_AA)

    y += 17
    cv2.putText(frame, "ATTENTION:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    if data.is_lateral_distracted:
        att_col = COLOR_AMBER
        att_lbl = f"LOOKING AWAY ({data.lateral_distraction_sec:.1f}s)"
    elif data.is_eye_darting:
        att_col = COLOR_CYAN
        att_lbl = "RESTLESS SCAN"
    else:
        att_col = COLOR_GREEN
        att_lbl = "FOCUSED / ENGAGED"
    cv2.putText(frame, att_lbl, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, att_col, 2, cv2.LINE_AA)

    y += 17
    cv2.putText(frame, "PUPIL DILATION:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    dil_str = f"{data.pupil_dilation_pct:+.1f}% ({data.pupil_state.split(' ')[0]})"
    dil_col = COLOR_CYAN if "EFFORT" in data.pupil_state else COLOR_WHITE
    cv2.putText(frame, dil_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, dil_col, 1, cv2.LINE_AA)

    y += 17
    cv2.putText(frame, "BLINK RATE:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    bpm_col = COLOR_AMBER if data.is_rapid_blinking else COLOR_GREEN
    bpm_tag = " (STRESS)" if data.is_rapid_blinking else " BPM"
    bpm_str = f"{data.blink_bpm:.0f}{bpm_tag}"
    cv2.putText(frame, bpm_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bpm_col, 2 if data.is_rapid_blinking else 1, cv2.LINE_AA)

    y += 8
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 3: Biomechanical Physical Metrics
    y += 17
    cv2.putText(frame, "EYE DISP (mm):", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    eye_str = f"dX: {data.metric_delta_x_mm:+.1f} | dY: {data.metric_delta_y_mm:+.1f}"
    cv2.putText(frame, eye_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)

    y += 17
    cv2.putText(frame, "HEAD POSE (deg):", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    head_str = f"Y:{data.head_yaw_deg:+.0f} P:{data.head_pitch_deg:+.0f}"
    cv2.putText(frame, head_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)

    y += 8
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 4: Dwell & Movement
    y += 17
    cv2.putText(frame, "DWELL TIME:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{data.time_in_zone:.1f}s", (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 2, cv2.LINE_AA)

    y += 17
    cv2.putText(frame, "MOVEMENT:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, data.movement_str, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)

    y += 17
    cv2.putText(frame, "SPEED:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    spd_col = COLOR_GREEN if data.movement_speed == "NORMAL" else (COLOR_AMBER if data.movement_speed == "FAST" else COLOR_TEXT_DIM)
    cv2.putText(frame, data.movement_speed, (x1 + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, spd_col, 2, cv2.LINE_AA)

    y += 8
    cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (45, 50, 60), 1)

    # Section 5: Recent History
    y += 16
    cv2.putText(frame, "RECENT VISITS:", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    if not data.history:
        y += 15
        cv2.putText(frame, "- Stating gaze history...", (x1 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    else:
        for ev in reversed(data.history[-2:]):
            y += 15
            txt = f"- [{ev.grid_coords[0]},{ev.grid_coords[1]}] {ev.dwell_sec:.1f}s at {ev.timestamp}"
            cv2.putText(frame, txt, (x1 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Footer Hotkeys
    y = y2 - 10
    cv2.putText(frame, "[f] Fullscreen | [v] View | [c] Calib | [q] Exit", (x1 + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_TEXT_DIM, 1, cv2.LINE_AA)


def draw_pip_camera(canvas, raw_frame, calibrator: GazeGridCalibrator, show_zoomed_eye: bool = True, pip_w: int = 310, pip_h: int = 145):
    """
    Renders picture-in-picture window in the bottom-right corner.
    Can display either:
    1. Binocular Dual-Eye Monitor (side-by-side magnified view of both eyes with pupil crosshairs & vectors)
    2. Full webcam feed with landmarks.
    """
    h_can, w_can, _ = canvas.shape
    x1 = w_can - pip_w - 14
    y1 = h_can - pip_h - 14
    x2 = x1 + pip_w
    y2 = y1 + pip_h

    if show_zoomed_eye and calibrator.last_eye1_crop is not None and calibrator.last_eye2_crop is not None:
        half_w = pip_w // 2

        # 1. Eye 1 (Viewer's Left Eye / Mirror Right)
        c1_h, c1_w, _ = calibrator.last_eye1_crop.shape
        eye1_view = cv2.resize(calibrator.last_eye1_crop, (half_w, pip_h), interpolation=cv2.INTER_LINEAR)
        sx1 = half_w / max(1.0, c1_w)
        sy1 = pip_h / max(1.0, c1_h)
        p1_x, p1_y = int(calibrator.last_pupil1_in_crop[0] * sx1), int(calibrator.last_pupil1_in_crop[1] * sy1)
        b1_x, b1_y = int(calibrator.last_base1_in_crop[0] * sx1), int(calibrator.last_base1_in_crop[1] * sy1)

        if calibrator.is_calibrated:
            cv2.line(eye1_view, (b1_x, b1_y), (p1_x, p1_y), (245, 195, 60), 2, cv2.LINE_AA)
            cv2.circle(eye1_view, (b1_x, b1_y), 4, (165, 170, 180), 1, cv2.LINE_AA)
        cv2.drawMarker(eye1_view, (p1_x, p1_y), (75, 225, 125), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        cv2.circle(eye1_view, (p1_x, p1_y), 4, (75, 225, 125), 1, cv2.LINE_AA)

        # 2. Eye 2 (Viewer's Right Eye / Mirror Left)
        c2_h, c2_w, _ = calibrator.last_eye2_crop.shape
        eye2_view = cv2.resize(calibrator.last_eye2_crop, (half_w, pip_h), interpolation=cv2.INTER_LINEAR)
        sx2 = half_w / max(1.0, c2_w)
        sy2 = pip_h / max(1.0, c2_h)
        p2_x, p2_y = int(calibrator.last_pupil2_in_crop[0] * sx2), int(calibrator.last_pupil2_in_crop[1] * sy2)
        b2_x, b2_y = int(calibrator.last_base2_in_crop[0] * sx2), int(calibrator.last_base2_in_crop[1] * sy2)

        if calibrator.is_calibrated:
            cv2.line(eye2_view, (b2_x, b2_y), (p2_x, p2_y), (245, 195, 60), 2, cv2.LINE_AA)
            cv2.circle(eye2_view, (b2_x, b2_y), 4, (165, 170, 180), 1, cv2.LINE_AA)
        cv2.drawMarker(eye2_view, (p2_x, p2_y), (75, 225, 125), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        cv2.circle(eye2_view, (p2_x, p2_y), 4, (75, 225, 125), 1, cv2.LINE_AA)

        # Combine into side-by-side dual-eye frame
        dual_pip = np.hstack([eye1_view, eye2_view])
        # Center vertical divider line
        cv2.line(dual_pip, (half_w, 0), (half_w, pip_h), (55, 60, 72), 2)

        canvas[y1:y2, x1:x2] = dual_pip
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (55, 60, 72), 1)

        # Top PIP Label
        cv2.rectangle(canvas, (x1, y1), (x1 + 185, y1 + 16), (20, 22, 28), -1)
        cv2.putText(canvas, "BINOCULAR ZOOM (2.5x HD)", (x1 + 4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.31, COLOR_CYAN, 1, cv2.LINE_AA)

        # Eye Labels
        cv2.putText(canvas, "L EYE", (x1 + 6, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "R EYE", (x1 + half_w + 6, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_WHITE, 1, cv2.LINE_AA)

        # Bottom live displacement readout
        cv2.rectangle(canvas, (x1, y2 - 18), (x2, y2), (20, 22, 28), -1)
        disp_txt = f"BINOCULAR dX:{calibrator.last_delta_mm[0]:+.1f}mm | dY:{calibrator.last_delta_mm[1]:+.1f}mm | DIL:{calibrator.smooth_dilation_pct:+.1f}%"
        cv2.putText(canvas, disp_txt, (x1 + 4, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_WHITE, 1, cv2.LINE_AA)

    else:
        # 2. Render Full Webcam Feed
        cam_pip = cv2.resize(raw_frame, (pip_w, pip_h))
        canvas[y1:y2, x1:x2] = cam_pip
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (55, 60, 72), 1)

        cv2.rectangle(canvas, (x1, y1), (x1 + 95, y1 + 16), (20, 22, 28), -1)
        cv2.putText(canvas, "WEBCAM FEED", (x1 + 4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.31, COLOR_GRAY, 1, cv2.LINE_AA)


def draw_grid_background(canvas, data: CalibratedGazeData, canvas_w: int, canvas_h: int):
    """Draws the full 4x3 screen grid and highlights the active focused zone."""
    cell_w = canvas_w / GRID_COLS
    cell_h = canvas_h / GRID_ROWS

    # 1. Fill cells and draw borders
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx1 = int(c * cell_w)
            cy1 = int(r * cell_h)
            cx2 = int((c + 1) * cell_w)
            cy2 = int((r + 1) * cell_h)

            is_active = (data.is_calibrated and data.active_grid == (r, c))

            if is_active:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (cx1 + 2, cy1 + 2), (cx2 - 2, cy2 - 2), COLOR_CELL_ACTIVE_BG, -1)
                cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)
                cv2.rectangle(canvas, (cx1 + 2, cy1 + 2), (cx2 - 2, cy2 - 2), COLOR_CELL_ACTIVE_BORDER, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_GRID_LINE, 1)

            # Cell Labels
            zone_key = (r, c)
            label = ZONE_LABELS.get(zone_key, "")
            label_col = COLOR_GREEN if is_active else COLOR_TEXT_DIM

            idx_str = f"[{r},{c}]"
            cv2.putText(canvas, idx_str, (cx1 + 12, cy1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, label_col, 1 if not is_active else 2, cv2.LINE_AA)

            cv2.putText(canvas, label, (cx1 + 12, cy1 + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, label_col, 1, cv2.LINE_AA)

            if is_active:
                dwell_tag = f"FOCUS: {data.time_in_zone:.1f}s"
                cv2.putText(canvas, dwell_tag, (cx1 + 12, cy2 - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

    # 2. Continuous Predicted Gaze Cursor
    if data.is_calibrated and data.face_detected:
        gx = int(data.smooth_gaze_norm[0] * canvas_w)
        gy = int(data.smooth_gaze_norm[1] * canvas_h)
        gx = max(10, min(canvas_w - 10, gx))
        gy = max(10, min(canvas_h - 10, gy))

        # Glowing target crosshair
        cv2.circle(canvas, (gx, gy), 14, (30, 35, 45), -1, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 10, COLOR_CELL_ACTIVE_BORDER, 2, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 3, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.line(canvas, (gx - 16, gy), (gx + 16, gy), COLOR_CELL_ACTIVE_BORDER, 1, cv2.LINE_AA)
        cv2.line(canvas, (gx, gy - 16), (gx, gy + 16), COLOR_CELL_ACTIVE_BORDER, 1, cv2.LINE_AA)

    # 3. Top Behavioral Alert Banner
    if data.is_calibrated and data.face_detected:
        if data.is_lateral_distracted:
            alert_w = 520
            alert_h = 36
            ax1 = (canvas_w - alert_w) // 2
            ay1 = 14
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (25, 20, 50), -1)
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), COLOR_AMBER, 2, cv2.LINE_AA)
            alert_txt = f"! DISTRACTION: LOOKING OFF-SCREEN ({data.lateral_distraction_sec:.1f}s)"
            cv2.putText(canvas, alert_txt, (ax1 + 14, ay1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_AMBER, 2, cv2.LINE_AA)
        elif data.is_eye_darting:
            alert_w = 420
            alert_h = 36
            ax1 = (canvas_w - alert_w) // 2
            ay1 = 14
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), (25, 30, 48), -1)
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + alert_w, ay1 + alert_h), COLOR_CYAN, 2, cv2.LINE_AA)
            alert_txt = "! RESTLESS GAZE: EYE DARTING DETECTED"
            cv2.putText(canvas, alert_txt, (ax1 + 14, ay1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_CYAN, 2, cv2.LINE_AA)


def draw_calibration_overlay(canvas, calibrator: GazeGridCalibrator, canvas_w: int, canvas_h: int):
    """
    Renders guided calibration overlay with Saccade-Safe settling ring
    and Phase indicator (Phase 1 Baseline Lock vs Phase 2 Snake Grid).
    """
    row, col, (tx, ty), progress, phase_label = calibrator.get_current_target_info(canvas_w, canvas_h)
    in_settle = calibrator.is_in_settle_window()

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_w, canvas_h), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Highlight active cell
    cell_w = canvas_w / GRID_COLS
    cell_h = canvas_h / GRID_ROWS
    bx1, by1 = int(col * cell_w), int(row * cell_h)
    bx2, by2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (40, 50, 70), 2, cv2.LINE_AA)

    # Top instructions banner
    banner_w = 640
    banner_h = 56
    bx = canvas_w // 2 - banner_w // 2
    cv2.rectangle(canvas, (bx, 14), (bx + banner_w, 14 + banner_h), (20, 24, 32), -1)
    cv2.rectangle(canvas, (bx, 14), (bx + banner_w, 14 + banner_h), (60, 70, 90), 1)

    cv2.putText(canvas, phase_label, (bx + 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 2, cv2.LINE_AA)
    subtext = "Settle eyes on target (locking fixation)..." if in_settle else "Recording clean fixation frames..."
    sub_col = COLOR_AMBER if in_settle else COLOR_GREEN
    cv2.putText(canvas, subtext, (bx + 20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sub_col, 1, cv2.LINE_AA)

    t = time.time()
    pulse_radius = int(16 + 5 * math.sin(t * PULSE_SPEED))
    target_radius = 28

    if in_settle:
        # Animated settling ring shrinking to center
        settle_frac = (t * 3.0) % 1.0
        ring_r = int(24 + 18 * (1.0 - settle_frac))
        cv2.circle(canvas, (tx, ty), ring_r, COLOR_AMBER, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Look Here", (tx - 32, ty + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_AMBER, 1, cv2.LINE_AA)
    else:
        # Green progress circle filling up as samples are recorded
        angle_end = int(progress * 360)
        cv2.ellipse(canvas, (tx, ty), (target_radius, target_radius), 0, -90, -90 + angle_end, COLOR_GREEN, 3, cv2.LINE_AA)
        cv2.ellipse(canvas, (tx, ty), (target_radius, target_radius), 0, -90 + angle_end, 270, (45, 50, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Zone [{row},{col}]", (tx - 36, ty + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)

    # Core target dot
    cv2.circle(canvas, (tx, ty), pulse_radius, (20, 25, 35), -1, cv2.LINE_AA)
    cv2.circle(canvas, (tx, ty), pulse_radius, COLOR_CYAN if in_settle else COLOR_GREEN, 2, cv2.LINE_AA)
    cv2.circle(canvas, (tx, ty), 4, COLOR_WHITE, -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Unable to connect to webcam at index {CAMERA_INDEX}.")
        return

    # Request High-Resolution 1280x720 capture for maximum iris pixel resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at {actual_cam_w}x{actual_cam_h} HD resolution.")

    # Initialize MediaPipe Face Mesh with iris refinement
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    calibrator = GazeGridCalibrator()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DEFAULT_CANVAS_W, DEFAULT_CANVAS_H)

    is_fullscreen = False
    show_zoomed_eye = True
    current_w = DEFAULT_CANVAS_W
    current_h = DEFAULT_CANVAS_H

    if ALWAYS_CALIBRATE_ON_START or not calibrator.is_calibrated:
        print("Starting Zoomed-In Eye Calibration sequence...")
        print("Step 1: Look at Center to lock eye socket baseline.")
        print("Step 2: Follow dots smoothly along the snake path.")
        calibrator.start_calibration()
    else:
        print("Zoomed Eye Calibration profile loaded. Eye tracking active.")
        print("Press 'f' for Fullscreen | 'v' to toggle Zoomed Eye View | 'c' to recalibrate | 'q' to quit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        # Flip horizontally for natural mirror interaction
        raw_frame = cv2.flip(raw_frame, 1)
        raw_h, raw_w, _ = raw_frame.shape

        # Process MediaPipe Face Mesh
        rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

        # Handle Fullscreen canvas sizing
        if is_fullscreen:
            rect = cv2.getWindowImageRect(WINDOW_TITLE)
            if rect[2] > 100 and rect[3] > 100:
                current_w = rect[2]
                current_h = rect[3]
        else:
            current_w = DEFAULT_CANVAS_W
            current_h = DEFAULT_CANVAS_H

        # Master display canvas
        canvas = np.full((current_h, current_w, 3), COLOR_BG, dtype=np.uint8)

        # 1. Calibration Routine Loop
        if calibrator.is_calibrating:
            feat_dict = calibrator.extract_features(landmarks, raw_frame, raw_w, raw_h) if landmarks else None
            finished = calibrator.record_calibration_frame(feat_dict, current_w, current_h)

            if finished:
                print("Zoomed Eye Calibration complete! Switching to live tracking.")
                gaze_data = calibrator.update(landmarks, raw_frame, raw_w, raw_h)
                draw_grid_background(canvas, gaze_data, current_w, current_h)
                draw_hud(canvas, gaze_data)
                draw_pip_camera(canvas, raw_frame, calibrator, show_zoomed_eye)
            else:
                draw_grid_background(canvas, CalibratedGazeData(is_calibrated=False), current_w, current_h)
                draw_calibration_overlay(canvas, calibrator, current_w, current_h)
                draw_pip_camera(canvas, raw_frame, calibrator, show_zoomed_eye)

        # 2. Live Tracking Loop
        else:
            gaze_data = calibrator.update(landmarks, raw_frame, raw_w, raw_h)
            draw_grid_background(canvas, gaze_data, current_w, current_h)
            draw_hud(canvas, gaze_data)
            draw_pip_camera(canvas, raw_frame, calibrator, show_zoomed_eye)

        cv2.imshow(WINDOW_TITLE, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
            break
        elif key in (ord('c'), ord('C')):    # 'c' to recalibrate
            print("Starting recalibration sequence...")
            calibrator.start_calibration()
        elif key in (ord('v'), ord('V')):    # 'v' to toggle Zoomed Eye vs Full Webcam
            show_zoomed_eye = not show_zoomed_eye
            view_name = "ZOOMED EYE (2.5x HD)" if show_zoomed_eye else "FULL WEBCAM"
            print(f"Picture-in-Picture switched to: {view_name}")
        elif key in (ord('f'), ord('F')):    # 'f' to toggle Fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Switched to FULLSCREEN mode (Maximum visual angle & accuracy).")
            else:
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_TITLE, DEFAULT_CANVAS_W, DEFAULT_CANVAS_H)
                print("Switched to WINDOWED mode.")

        # Check if window was closed via [X]
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
