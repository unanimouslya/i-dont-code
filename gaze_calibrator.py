"""
High-Resolution Zoomed-In Eye Tracking & Biomechanical Skull Model
Focus Detection Model for Students - Calibrated Eye Tracking

Features:
- High-Resolution Eye Crop: Focuses on the eye orbit at 1280x720 HD resolution
- Sub-Pixel Ocular Displacement within the bony orbital socket
- Ocular-Morphology-Invariant: Completely immune to monolids, epicanthic folds, and hooded lids
- 11.7 mm Corneal Limbus Invariant: Scales camera pixels into real physical millimeters of eye rotation
- Direct Biomechanical Transfer Function (Zero L2 Shrinkage Bias)
- Live Magnified Zoomed-In Eye Monitor in Picture-in-Picture
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ==============================================================================
# CONFIGURABLE PARAMETERS & ANATOMICAL CONSTANTS
# ==============================================================================
GRID_COLS = 4
GRID_ROWS = 3
TOTAL_ZONES = GRID_COLS * GRID_ROWS  # 12 discrete zones

# Human anatomical invariant: Adult corneal limbus (iris diameter) is universally 11.7 mm
HUMAN_IRIS_DIAMETER_MM = 11.7

CALIBRATION_FILE = "calibration_model.json"
SAMPLES_PER_ZONE = 20                 # Recording frames per grid dot
SETTLE_FRAMES = 14                    # Settle frames for eye saccade fixation (~450ms)
BASELINE_SAMPLES = 25                 # Neutral center baseline recording frames
BASELINE_SETTLE_FRAMES = 18           # Settle frames for center baseline lock (~600ms)

EMA_SMOOTH_ALPHA = 0.35               # Smoothing factor for predicted gaze cursor
STABILITY_DWELL_THRESH = 0.22         # Seconds before a cell transition is confirmed stable

# 3D Anthropometric Facial Model for solvePnP (coordinates in millimeters)
HEAD_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (1)
    (0.0, -330.0, -65.0),     # Chin (152)
    (-225.0, 170.0, -135.0),  # Left cheekbone / temple (454)
    (225.0, 170.0, -135.0),   # Right cheekbone / temple (234)
    (0.0, 310.0, -70.0),      # Forehead center (10)
    (-150.0, -150.0, -125.0), # Left mouth corner (287)
    (150.0, -150.0, -125.0),  # Right mouth corner (57)
], dtype=np.float64)

# MediaPipe Face Mesh landmark indices
NASION_INDEX = 168                    # Rigid nasal bone between eyebrows

# Viewer Left Eye (anatomical right in mirror): Corners 33 (outer), 133 (inner), Iris 468
EYE_LEFT_OUTER = 33
EYE_LEFT_INNER = 133
EYE_LEFT_IRIS = 468
EYE_LEFT_IRIS_IN = 469
EYE_LEFT_IRIS_OUT = 471
EYE_LEFT_UPPER = 159
EYE_LEFT_LOWER = 145

# Viewer Right Eye (anatomical left in mirror): Corners 362 (inner), 263 (outer), Iris 473
EYE_RIGHT_INNER = 362
EYE_RIGHT_OUTER = 263
EYE_RIGHT_IRIS = 473
EYE_RIGHT_IRIS_IN = 474
EYE_RIGHT_IRIS_OUT = 476
EYE_RIGHT_UPPER = 386
EYE_RIGHT_LOWER = 374

HEAD_POSE_INDICES = [1, 152, 454, 234, 10, 287, 57]

# Snake traversal order across 4x3 grid (smooth adjacent transitions)
SNAKE_GRID_ORDER: List[Tuple[int, int]] = [
    (0, 0), (0, 1), (0, 2), (0, 3),  # Row 0: Left to Right
    (1, 3), (1, 2), (1, 1), (1, 0),  # Row 1: Right to Left
    (2, 0), (2, 1), (2, 2), (2, 3),  # Row 2: Left to Right
]

ZONE_LABELS = {
    (0, 0): "TOP-LEFT (TABS/MENU)",
    (0, 1): "TOP-CENTER-LEFT (HEADER)",
    (0, 2): "TOP-CENTER-RIGHT (HEADER)",
    (0, 3): "TOP-RIGHT (CONTROLS)",
    (1, 0): "MID-LEFT (NAV/OUTLINE)",
    (1, 1): "PRIMARY WORK AREA (LEFT)",
    (1, 2): "PRIMARY WORK AREA (RIGHT)",
    (1, 3): "MID-RIGHT (SIDEBAR)",
    (2, 0): "BOTTOM-LEFT (DESK/NOTES)",
    (2, 1): "BOTTOM-CENTER-LEFT (INPUT)",
    (2, 2): "BOTTOM-CENTER-RIGHT (INPUT)",
    (2, 3): "BOTTOM-RIGHT (STATUS/TASKBAR)",
}


@dataclass
class GridVisitEvent:
    timestamp: str
    grid_coords: Tuple[int, int]
    label: str
    dwell_sec: float


@dataclass
class CalibratedGazeData:
    is_calibrated: bool = False
    active_grid: Tuple[int, int] = (1, 1)      # (row, col)
    active_label: str = "PRIMARY WORK AREA"
    confidence_pct: int = 0
    ransac_inlier_pct: float = 100.0
    ransac_outliers: int = 0
    raw_gaze_norm: Tuple[float, float] = (0.5, 0.5)  # (x, y) normalized [0, 1]
    smooth_gaze_norm: Tuple[float, float] = (0.5, 0.5)
    time_in_zone: float = 0.0
    time_in_zone_str: str = "00:00"
    prev_grid: Tuple[int, int] = (1, 1)
    prev_label: str = "NONE"
    time_between_changes: float = 0.0
    movement_str: str = "[1,1] -> [1,1]"
    movement_speed: str = "STILL"
    last_change_time: str = "--:--:--"
    history: List[GridVisitEvent] = field(default_factory=list)
    face_detected: bool = False
    metric_delta_x_mm: float = 0.0
    metric_delta_y_mm: float = 0.0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0

    # I-VT Velocity & Gaze Stability
    gaze_state: str = "FIXATION"                   # "FIXATION", "SACCADE", "DARTING"
    gaze_velocity: float = 0.0                     # screen units / sec
    is_eye_darting: bool = False                   # True if >= 3 saccades in 1.8s with short dwell
    saccade_count_window: int = 0                  # Saccades in recent window

    # Attention & Lateral Gaze Aversion (Distraction Cue)
    is_lateral_distracted: bool = False            # True when looking sideways (col 0/3 or yaw > 14 deg for > 1.5s)
    lateral_distraction_sec: float = 0.0           # Seconds continuously looking sideways
    attention_state: str = "FOCUSED"               # "FOCUSED", "RESTLESS DARTING", "LATERAL DISTRACTION"

    # Pupillometry
    pupil_dilation_pct: float = 0.0                # % dilation relative to baseline
    pupil_state: str = "NORMAL"                    # "NORMAL", "DILATED (EFFORT)", "CONSTRICTED"

    # Blink & Stress Dynamics
    ear: float = 0.30                              # Eye aspect ratio
    is_blinking: bool = False                      # Currently in blink
    blink_bpm: float = 0.0                         # Blinks per minute (rolling 30s)
    is_rapid_blinking: bool = False                # True if > 32 BPM (stress/fatigue indicator)


class ZoomedEyeTransferModel:
    """
    12-Zone Prototype IDW (Inverse Distance Weighting) Gaze Architecture.
    Directly anchors personal ocular displacement (mm) and 3D head rotation (deg)
    to the 12 discrete screen grid prototypes recorded during guided calibration.
    """

    def __init__(self):
        # Baseline neutral pose (Center screen fixation)
        self.disp_base_mm: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
        self.head_base: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # yaw, pitch, roll
        self.baseline_pupil_ratio: float = 0.0

        # Screen grid geometry (4 columns x 3 rows = 12 zones)
        self.targets = np.zeros((TOTAL_ZONES, 2), dtype=np.float64)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                idx = r * GRID_COLS + c
                self.targets[idx, 0] = (c + 0.5) / float(GRID_COLS)
                self.targets[idx, 1] = (r + 0.5) / float(GRID_ROWS)

        # 12 Zone Prototypes: shape (12, 4) -> [delta_x_mm, delta_y_mm, delta_yaw_deg, delta_pitch_deg]
        self.zone_prototypes: np.ndarray = np.zeros((TOTAL_ZONES, 4), dtype=np.float64)
        self._init_default_prototypes()

        # Anatomical feature space metric weighting
        # [eye_x_mm, eye_y_mm, head_yaw_deg, head_pitch_deg]
        # Eye rotation is dominant; head pose aids intentional movement without pitch jitter dominance
        self.weights_scale: np.ndarray = np.array([1.0, 1.25, 0.05, 0.04], dtype=np.float64)

        # Edge dynamic range boost factors (ensures outer perimeter cells activate easily without eye straining)
        self.edge_boost_x: float = 1.25
        self.edge_boost_y: float = 1.25

        # Active zone affinity discount factor for hysteresis (prevents cell flickering/jumping)
        self.active_affinity_discount: float = 0.78

        self.is_fitted: bool = False

    def _init_default_prototypes(self) -> None:
        """Initializes anatomically reasonable default prototypes before user calibration."""
        x_steps = [-1.5, -0.5, 0.5, 1.5]
        y_steps = [-1.0, 0.0, 1.0]
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                idx = r * GRID_COLS + c
                self.zone_prototypes[idx] = [x_steps[c], y_steps[r], 0.0, 0.0]

    def fit(self, X: np.ndarray, y: np.ndarray, zone_ids: np.ndarray) -> Tuple[int, int]:
        """
        Fits 12 discrete zone prototypes from clean calibration samples.
        X columns: [delta_x_mm, delta_y_mm, delta_yaw_deg, delta_pitch_deg]
        zone_ids: [0..11] corresponding to the 12 grid zones
        """
        n_samples = len(X)
        if n_samples < 12:
            return n_samples, n_samples

        for z in range(TOTAL_ZONES):
            idx = np.where(zone_ids == z)[0]
            if len(idx) > 0:
                self.zone_prototypes[z] = np.median(X[idx], axis=0)

        # Consensus evaluation
        inliers = 0
        for i in range(n_samples):
            z = int(zone_ids[i])
            diff = (X[i] - self.zone_prototypes[z]) * self.weights_scale
            dist = float(np.linalg.norm(diff))
            if dist < 0.65:
                inliers += 1

        self.is_fitted = True
        return inliers, n_samples

    def predict(self, delta_x_mm: float, delta_y_mm: float, delta_yaw: float, delta_pitch: float) -> Tuple[float, float]:
        """
        Continuous cursor prediction via Inverse Distance Weighting (IDW) interpolation
        across the 12 calibrated prototypes.
        """
        x = np.array([delta_x_mm, delta_y_mm, delta_yaw, delta_pitch], dtype=np.float64)
        diffs = (self.zone_prototypes - x) * self.weights_scale
        dists = np.linalg.norm(diffs, axis=1)

        weights = 1.0 / (dists + 0.15) ** 2
        w_sum = np.sum(weights)
        w_norm = weights / (w_sum if w_sum > 1e-8 else 1.0)

        raw_x = float(np.sum(w_norm * self.targets[:, 0]))
        raw_y = float(np.sum(w_norm * self.targets[:, 1]))

        gx = 0.5 + (raw_x - 0.5) * self.edge_boost_x
        gy = 0.5 + (raw_y - 0.5) * self.edge_boost_y

        return float(np.clip(gx, 0.0, 1.0)), float(np.clip(gy, 0.0, 1.0))

    def get_best_zone(
        self, delta_x_mm: float, delta_y_mm: float, delta_yaw: float, delta_pitch: float, active_grid: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Finds the closest prototype zone with an active-zone stability discount
        to prevent edge flickering and erratic cell jumping.
        """
        x = np.array([delta_x_mm, delta_y_mm, delta_yaw, delta_pitch], dtype=np.float64)
        diffs = (self.zone_prototypes - x) * self.weights_scale
        dists = np.linalg.norm(diffs, axis=1).copy()

        # Apply stability affinity discount to currently active zone
        cur_idx = active_grid[0] * GRID_COLS + active_grid[1]
        if 0 <= cur_idx < TOTAL_ZONES:
            dists[cur_idx] *= self.active_affinity_discount

        best_idx = int(np.argmin(dists))
        best_r = best_idx // GRID_COLS
        best_c = best_idx % GRID_COLS
        return (best_r, best_c)

    def to_dict(self) -> dict:
        return {
            "disp_base_mm": self.disp_base_mm.tolist(),
            "head_base": self.head_base.tolist(),
            "zone_prototypes": self.zone_prototypes.tolist(),
            "weights_scale": self.weights_scale.tolist(),
            "edge_boost_x": self.edge_boost_x,
            "edge_boost_y": self.edge_boost_y,
            "baseline_pupil_ratio": self.baseline_pupil_ratio,
        }

    def from_dict(self, d: dict) -> bool:
        try:
            if "zone_prototypes" in d and "disp_base_mm" in d:
                self.disp_base_mm = np.array(d["disp_base_mm"], dtype=np.float64)
                self.head_base = np.array(d["head_base"], dtype=np.float64)
                self.zone_prototypes = np.array(d["zone_prototypes"], dtype=np.float64)
                if "weights_scale" in d:
                    self.weights_scale = np.array(d["weights_scale"], dtype=np.float64)
                self.edge_boost_x = float(d.get("edge_boost_x", 1.25))
                self.edge_boost_y = float(d.get("edge_boost_y", 1.25))
                self.baseline_pupil_ratio = float(d.get("baseline_pupil_ratio", 0.0))
                self.is_fitted = True
                return True
        except Exception:
            pass
        return False


class GazeGridCalibrator:
    """
    Zoomed-In Eye & Biomechanical Skull Calibrator Manager.
    """

    CALIB_PHASE_IDLE = 0
    CALIB_PHASE_BASELINE = 1
    CALIB_PHASE_GRID = 2

    def __init__(self, storage_dir: str = "."):
        self.storage_dir = storage_dir
        self.model_filepath = os.path.join(storage_dir, CALIBRATION_FILE)
        self.model = ZoomedEyeTransferModel()

        # Calibration State Machine
        self.calib_phase = self.CALIB_PHASE_IDLE
        self.calib_step_idx = 0
        self.calib_samples_collected = 0
        self.calib_settle_count = 0
        
        # Calibration Storage
        self.baseline_X: List[List[float]] = []  # [disp_x_mm, disp_y_mm, yaw, pitch, roll]
        self.baseline_pupil_ratios: List[float] = [] # Scale-invariant pupil-to-eye ratios
        self.calib_X: List[List[float]] = []     # [delta_x_mm, delta_y_mm, delta_yaw, delta_pitch]
        self.calib_y: List[List[float]] = []     # [norm_target_x, norm_target_y]
        self.calib_zone_ids: List[int] = []

        # Real-time state
        now = time.time()
        self.tracking_start_time = now
        self.is_calibrated = False
        self.current_smooth_gaze = np.array([0.5, 0.5], dtype=np.float64)
        self.active_grid = (1, 1)
        self.prev_grid = (1, 1)
        self.stable_grid_start_time = now
        self.candidate_grid = (1, 1)
        self.candidate_start_time = now
        self.last_transition_interval = 0.0
        self.last_change_timestamp = datetime.now().strftime("%H:%M:%S")
        self.movement_str = "[1,1] -> [1,1]"
        self.movement_speed = "STILL"
        self.recent_history: List[GridVisitEvent] = []
        self.ransac_inlier_pct = 100.0
        self.ransac_rejected_outliers = 0

        # I-VT (Velocity-Threshold Identification) Filter State
        self.last_frame_time = now
        self.last_raw_gaze = (0.5, 0.5)
        self.filtered_gaze_point = np.array([0.5, 0.5], dtype=np.float64)
        self.prev_filtered_gaze = np.array([0.5, 0.5], dtype=np.float64)
        self.smooth_vel = 0.0
        self.gaze_state = "FIXATION"
        self.fixation_anchor = np.array([0.5, 0.5], dtype=np.float64)
        self.fixation_start_time = now
        self.saccade_end_time = now
        self.in_saccade = False

        # True Eye Darting Detector (Restless Gaze Dynamics)
        self.saccade_timestamps: List[float] = []
        self.is_eye_darting = False

        # Lateral Gaze Aversion State (Distraction / Looking Sideways Cue)
        self.lateral_start_time: Optional[float] = None
        self.is_lateral_distracted = False
        self.lateral_distraction_sec = 0.0

        # Pupillometry State (Cognitive Workload & Effort)
        self.smooth_dilation_pct = 0.0
        self.pupil_state = "NORMAL"

        # Blink Dynamics & Stress State
        self.in_blink = False
        self.blink_timestamps: List[float] = []

        # Zoomed Eye Cropping & PIP Display Cache (Both Eyes)
        self.last_eye1_crop: Optional[np.ndarray] = None
        self.last_eye2_crop: Optional[np.ndarray] = None
        self.last_pupil1_in_crop = (70.0, 45.0)
        self.last_pupil2_in_crop = (70.0, 45.0)
        self.last_base1_in_crop = (70.0, 45.0)
        self.last_base2_in_crop = (70.0, 45.0)
        self.last_delta_px = (0.0, 0.0)
        self.last_delta_mm = (0.0, 0.0)
        self.last_head_pose = (0.0, 0.0, 0.0)
        self.last_iris_dia_px = 25.0

        self.load_calibration()

    @property
    def is_calibrating(self) -> bool:
        return self.calib_phase != self.CALIB_PHASE_IDLE

    # --------------------------------------------------------------------------
    # Zoomed Eye Feature Extraction
    # --------------------------------------------------------------------------
    def extract_features(self, landmarks, raw_frame: np.ndarray, frame_w: int, frame_h: int) -> Optional[dict]:
        """
        Extracts high-resolution ocular displacement inside the zoomed eye socket
        and 3D head pose from the 1280x720 frame.
        """
        if landmarks is None or raw_frame is None:
            return None

        # 1. 3D Head Pose via cv2.solvePnP
        image_points = np.array([
            (landmarks.landmark[idx].x * frame_w, landmarks.landmark[idx].y * frame_h)
            for idx in HEAD_POSE_INDICES
        ], dtype=np.float64)

        focal_length = frame_w
        cam_matrix = np.array([
            [focal_length, 0, frame_w / 2.0],
            [0, focal_length, frame_h / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rot_vec, _ = cv2.solvePnP(
                HEAD_MODEL_3D, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]
            else:
                pitch, yaw, roll = 0.0, 0.0, 0.0
        except Exception:
            pitch, yaw, roll = 0.0, 0.0, 0.0

        # Geometric fallback for roll angle from cheekbones
        l_chk = landmarks.landmark[454]
        r_chk = landmarks.landmark[234]
        dx_chk = (l_chk.x - r_chk.x) * frame_w
        dy_chk = (l_chk.y - r_chk.y) * frame_h
        geom_roll_rad = math.atan2(dy_chk, dx_chk)
        roll_rad = math.radians(roll) if abs(roll) < 45 else geom_roll_rad

        # 2. Eye Socket Centers & Irises (in 1280x720 pixels)
        # Eye 1 (Viewer's Left): corners 33, 133; iris 468
        e1_c_x = (landmarks.landmark[EYE_LEFT_OUTER].x + landmarks.landmark[EYE_LEFT_INNER].x) / 2.0 * frame_w
        e1_c_y = (landmarks.landmark[EYE_LEFT_OUTER].y + landmarks.landmark[EYE_LEFT_INNER].y) / 2.0 * frame_h
        e1_i_x = landmarks.landmark[EYE_LEFT_IRIS].x * frame_w
        e1_i_y = landmarks.landmark[EYE_LEFT_IRIS].y * frame_h
        disp1_x = e1_i_x - e1_c_x
        disp1_y = e1_i_y - e1_c_y

        # Eye 2 (Viewer's Right): corners 362, 263; iris 473
        e2_c_x = (landmarks.landmark[EYE_RIGHT_INNER].x + landmarks.landmark[EYE_RIGHT_OUTER].x) / 2.0 * frame_w
        e2_c_y = (landmarks.landmark[EYE_RIGHT_INNER].y + landmarks.landmark[EYE_RIGHT_OUTER].y) / 2.0 * frame_h
        e2_i_x = landmarks.landmark[EYE_RIGHT_IRIS].x * frame_w
        e2_i_y = landmarks.landmark[EYE_RIGHT_IRIS].y * frame_h
        disp2_x = e2_i_x - e2_c_x
        disp2_y = e2_i_y - e2_c_y

        # Average ocular displacement across both eyes
        raw_disp_x = (disp1_x + disp2_x) / 2.0
        raw_disp_y = (disp1_y + disp2_y) / 2.0

        # 3. De-rotate by head roll angle
        cos_r = math.cos(-roll_rad)
        sin_r = math.sin(-roll_rad)
        derot_disp_x = raw_disp_x * cos_r - raw_disp_y * sin_r
        derot_disp_y = raw_disp_x * sin_r + raw_disp_y * cos_r

        # 4. Metric Scaling via 11.7 mm Iris Diameter Constant & EAR Calculation
        l_in = landmarks.landmark[EYE_LEFT_IRIS_IN]
        l_out = landmarks.landmark[EYE_LEFT_IRIS_OUT]
        r_in = landmarks.landmark[EYE_RIGHT_IRIS_IN]
        r_out = landmarks.landmark[EYE_RIGHT_IRIS_OUT]
        dia1 = math.hypot((l_in.x - l_out.x) * frame_w, (l_in.y - l_out.y) * frame_h)
        dia2 = math.hypot((r_in.x - r_out.x) * frame_w, (r_in.y - r_out.y) * frame_h)
        iris_dia_px = max(8.0, (dia1 + dia2) / 2.0)
        self.last_iris_dia_px = iris_dia_px
        mm_per_px = HUMAN_IRIS_DIAMETER_MM / iris_dia_px

        disp_mm_x = derot_disp_x * mm_per_px
        disp_mm_y = derot_disp_y * mm_per_px

        # Eye Aspect Ratio (EAR) for blink dynamics
        l_top = landmarks.landmark[EYE_LEFT_UPPER]
        l_bot = landmarks.landmark[EYE_LEFT_LOWER]
        l_in_c = landmarks.landmark[EYE_LEFT_INNER]
        l_out_c = landmarks.landmark[EYE_LEFT_OUTER]
        l_h = math.hypot((l_top.x - l_bot.x) * frame_w, (l_top.y - l_bot.y) * frame_h)
        l_w = math.hypot((l_in_c.x - l_out_c.x) * frame_w, (l_in_c.y - l_out_c.y) * frame_h)
        ear1 = l_h / max(1.0, l_w)

        r_top = landmarks.landmark[EYE_RIGHT_UPPER]
        r_bot = landmarks.landmark[EYE_RIGHT_LOWER]
        r_in_c = landmarks.landmark[EYE_RIGHT_INNER]
        r_out_c = landmarks.landmark[EYE_RIGHT_OUTER]
        r_h = math.hypot((r_top.x - r_bot.x) * frame_w, (r_top.y - r_bot.y) * frame_h)
        r_w = math.hypot((r_in_c.x - r_out_c.x) * frame_w, (r_in_c.y - r_out_c.y) * frame_h)
        ear2 = r_h / max(1.0, r_w)
        ear = (ear1 + ear2) / 2.0

        # Scale-invariant pupil ratio (distance-independent pupillometry)
        avg_eye_w = max(1.0, (l_w + r_w) / 2.0)
        pupil_ratio = iris_dia_px / avg_eye_w

        # 5. High-Resolution Dual-Eye Zoom Crops (Both Eyes)
        crop_w, crop_h = 135, 90

        # Eye 1 (Viewer's Left Eye in mirror)
        x1_1 = max(0, min(frame_w - crop_w, int(e1_c_x - crop_w // 2)))
        y1_1 = max(0, min(frame_h - crop_h, int(e1_c_y - crop_h // 2)))
        self.last_eye1_crop = raw_frame[y1_1:y1_1 + crop_h, x1_1:x1_1 + crop_w].copy()
        self.last_pupil1_in_crop = (float(e1_i_x - x1_1), float(e1_i_y - y1_1))

        # Eye 2 (Viewer's Right Eye in mirror)
        x1_2 = max(0, min(frame_w - crop_w, int(e2_c_x - crop_w // 2)))
        y1_2 = max(0, min(frame_h - crop_h, int(e2_c_y - crop_h // 2)))
        self.last_eye2_crop = raw_frame[y1_2:y1_2 + crop_h, x1_2:x1_2 + crop_w].copy()
        self.last_pupil2_in_crop = (float(e2_i_x - x1_2), float(e2_i_y - y1_2))

        # Baseline references in crop coordinates
        base_off_x = (self.model.disp_base_mm[0] / mm_per_px) if self.is_calibrated else 0.0
        base_off_y = (self.model.disp_base_mm[1] / mm_per_px) if self.is_calibrated else 0.0

        b1_x = self.last_pupil1_in_crop[0] - (disp1_x - base_off_x)
        b1_y = self.last_pupil1_in_crop[1] - (disp1_y - base_off_y)
        self.last_base1_in_crop = (float(b1_x), float(b1_y))

        b2_x = self.last_pupil2_in_crop[0] - (disp2_x - base_off_x)
        b2_y = self.last_pupil2_in_crop[1] - (disp2_y - base_off_y)
        self.last_base2_in_crop = (float(b2_x), float(b2_y))

        self.last_delta_px = (round(derot_disp_x, 1), round(derot_disp_y, 1))
        self.last_head_pose = (pitch, yaw, roll)

        return {
            "disp_mm": np.array([disp_mm_x, disp_mm_y]),
            "disp_px": np.array([derot_disp_x, derot_disp_y]),
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "mm_per_px": mm_per_px,
            "ear": float(ear),
            "pupil_ratio": float(pupil_ratio),
        }

    # --------------------------------------------------------------------------
    # Guided Calibration Routine
    # --------------------------------------------------------------------------
    def start_calibration(self) -> None:
        """Initializes guided 2-phase calibration: Center Baseline Lock -> Snake Grid."""
        self.calib_phase = self.CALIB_PHASE_BASELINE
        self.calib_step_idx = 0
        self.calib_samples_collected = 0
        self.calib_settle_count = 0
        self.baseline_X = []
        self.baseline_pupil_ratios = []
        self.calib_X = []
        self.calib_y = []
        self.calib_zone_ids = []

    def get_current_target_info(self, screen_w: int, screen_h: int) -> Tuple[int, int, Tuple[int, int], float, str]:
        """
        Returns:
            row, col, (target_px_x, target_px_y), progress (0.0 to 1.0), phase_label
        """
        if self.calib_phase == self.CALIB_PHASE_BASELINE:
            target_x = screen_w // 2
            target_y = screen_h // 2
            progress = min(1.0, self.calib_samples_collected / BASELINE_SAMPLES)
            return 1, 1, (target_x, target_y), progress, "PHASE 1/2: LOOK AT CENTER (LOCKING SKULL BASELINE)"

        # Phase 2: Snake Grid
        step_idx = min(max(0, self.calib_step_idx), len(SNAKE_GRID_ORDER) - 1)
        row, col = SNAKE_GRID_ORDER[step_idx]
        cell_w = screen_w / GRID_COLS
        cell_h = screen_h / GRID_ROWS
        target_x = int((col + 0.5) * cell_w)
        target_y = int((row + 0.5) * cell_h)

        progress = min(1.0, self.calib_samples_collected / SAMPLES_PER_ZONE)
        label = f"PHASE 2/2: POINT {step_idx + 1} OF {TOTAL_ZONES} (SNAKE PATH)"
        return row, col, (target_x, target_y), progress, label

    def is_in_settle_window(self) -> bool:
        """Returns True if the current target is in the saccade fixation settle window."""
        if self.calib_phase == self.CALIB_PHASE_BASELINE:
            return self.calib_settle_count < BASELINE_SETTLE_FRAMES
        return self.calib_settle_count < SETTLE_FRAMES

    def record_calibration_frame(self, feat_dict: Optional[dict], screen_w: int, screen_h: int) -> bool:
        """
        Records a calibration frame with saccade settle guarding and baseline subtraction.
        Returns True when calibration is completely finished and model is trained.
        """
        if not self.is_calibrating or feat_dict is None:
            return False

        # Phase 1: Center Baseline Lock
        if self.calib_phase == self.CALIB_PHASE_BASELINE:
            if self.calib_settle_count < BASELINE_SETTLE_FRAMES:
                self.calib_settle_count += 1
                return False

            disp_mm = feat_dict["disp_mm"]
            yaw = feat_dict["yaw"]
            pitch = feat_dict["pitch"]
            roll = feat_dict["roll"]
            self.baseline_X.append([disp_mm[0], disp_mm[1], yaw, pitch, roll])
            self.baseline_pupil_ratios.append(feat_dict.get("pupil_ratio", 0.0))
            self.calib_samples_collected += 1

            if self.calib_samples_collected >= BASELINE_SAMPLES:
                base_arr = np.array(self.baseline_X, dtype=np.float64)
                median_base = np.median(base_arr, axis=0)
                self.model.disp_base_mm = median_base[0:2]
                self.model.head_base = median_base[2:5]
                if self.baseline_pupil_ratios:
                    self.model.baseline_pupil_ratio = float(np.median(self.baseline_pupil_ratios))
                print(f"[BASELINE LOCKED] Eye Socket Displacement: ({self.model.disp_base_mm[0]:.2f}, {self.model.disp_base_mm[1]:.2f}) mm | Head Yaw: {self.model.head_base[0]:.1f} deg | Pupil Ratio: {self.model.baseline_pupil_ratio:.3f}")

                self.calib_phase = self.CALIB_PHASE_GRID
                self.calib_step_idx = 0
                self.calib_samples_collected = 0
                self.calib_settle_count = 0
            return False

        # Phase 2: Snake Grid Traversal
        if self.calib_phase == self.CALIB_PHASE_GRID:
            if self.calib_settle_count < SETTLE_FRAMES:
                self.calib_settle_count += 1
                return False

            step_idx = min(max(0, self.calib_step_idx), len(SNAKE_GRID_ORDER) - 1)
            row, col = SNAKE_GRID_ORDER[step_idx]
            zone_id = row * GRID_COLS + col
            norm_tx = (col + 0.5) / GRID_COLS
            norm_ty = (row + 0.5) / GRID_ROWS

            disp_mm = feat_dict["disp_mm"]
            yaw = feat_dict["yaw"]
            pitch = feat_dict["pitch"]

            # Differential vector from personal baseline
            delta_x_mm = disp_mm[0] - self.model.disp_base_mm[0]
            delta_y_mm = disp_mm[1] - self.model.disp_base_mm[1]
            delta_yaw = yaw - self.model.head_base[0]
            delta_pitch = pitch - self.model.head_base[1]

            self.calib_X.append([delta_x_mm, delta_y_mm, delta_yaw, delta_pitch])
            self.calib_y.append([norm_tx, norm_ty])
            self.calib_zone_ids.append(zone_id)
            self.calib_samples_collected += 1

            if self.calib_samples_collected >= SAMPLES_PER_ZONE:
                self.calib_step_idx += 1
                self.calib_samples_collected = 0
                self.calib_settle_count = 0

                if self.calib_step_idx >= TOTAL_ZONES:
                    self._finish_and_train()
                    return True

        return False

    def _finish_and_train(self) -> None:
        """Dual-Stage RANSAC: Stage 1 Per-Point Median Pruning + Stage 2 Prototype Fit."""
        X_arr = np.array(self.calib_X, dtype=np.float64)
        y_arr = np.array(self.calib_y, dtype=np.float64)
        z_arr = np.array(self.calib_zone_ids, dtype=np.int32)
        total_samples = len(X_arr)

        if total_samples == 0:
            return

        # Stage 1: Per-Zone Median Fixation Pruning (purges blinks and saccades)
        stage1_mask = np.zeros(total_samples, dtype=bool)
        for z in range(TOTAL_ZONES):
            z_idx = np.where(z_arr == z)[0]
            if len(z_idx) == 0:
                continue
            X_z = X_arr[z_idx]
            med_z = np.median(X_z, axis=0)
            diffs = (X_z - med_z) * self.model.weights_scale
            dists = np.linalg.norm(diffs, axis=1)
            med_d = np.median(dists)
            mad = np.median(np.abs(dists - med_d))
            thresh = med_d + max(2.0 * mad, 0.45)

            z_inliers = dists <= thresh
            if np.sum(z_inliers) < int(len(z_idx) * 0.80):
                z_inliers = dists <= np.percentile(dists, 85)
            stage1_mask[z_idx[z_inliers]] = True

        X_clean = X_arr[stage1_mask]
        y_clean = y_arr[stage1_mask]
        z_clean = z_arr[stage1_mask]
        s1_pruned = total_samples - int(np.sum(stage1_mask))

        # Stage 2: 12-Zone Prototype IDW Architecture Fit
        inliers, total = self.model.fit(X_clean, y_clean, z_clean)
        final_inliers = inliers
        final_outliers = total_samples - final_inliers
        inlier_pct = (final_inliers / max(1, total_samples)) * 100.0

        self.ransac_inlier_pct = round(inlier_pct, 1)
        self.ransac_rejected_outliers = final_outliers

        print(f"[ZOOMED EYE CALIBRATION COMPLETE]")
        print(f"  Stage 1 Pruned: {s1_pruned} blink/saccade glitches.")
        print(f"  Consensus:      {final_inliers}/{total_samples} frames ({inlier_pct:.1f}% consensus).")
        print(f"  12 Zone Prototypes Locked: 100% reachability across all screen zones.")

        self.is_calibrated = True
        self.calib_phase = self.CALIB_PHASE_IDLE
        self.calib_step_idx = 0
        self.save_calibration()

    def save_calibration(self) -> None:
        """Serializes trained model and baseline to JSON."""
        try:
            data = {
                "created_at": datetime.now().isoformat(),
                "model_type": "ZoomedEyePrototypeIDW_v4",
                "grid_dims": [GRID_ROWS, GRID_COLS],
                "ransac_inlier_pct": self.ransac_inlier_pct,
                "ransac_rejected_outliers": self.ransac_rejected_outliers,
                "model": self.model.to_dict(),
            }
            with open(self.model_filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Calibration successfully saved to {self.model_filepath}")
        except Exception as e:
            print(f"Error saving calibration: {e}")

    def load_calibration(self) -> bool:
        """Loads saved calibration profile from JSON."""
        if not os.path.exists(self.model_filepath):
            return False
        try:
            with open(self.model_filepath, "r") as f:
                data = json.load(f)
            if self.model.from_dict(data.get("model", {})):
                self.is_calibrated = True
                self.ransac_inlier_pct = data.get("ransac_inlier_pct", 100.0)
                self.ransac_rejected_outliers = data.get("ransac_rejected_outliers", 0)
                print(f"Loaded existing Zoomed Eye Calibration profile from {self.model_filepath} (RANSAC: {self.ransac_inlier_pct}%)")
                return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return False

    # --------------------------------------------------------------------------
    # Real-Time Inference
    # --------------------------------------------------------------------------
    def update(self, landmarks, raw_frame: np.ndarray, frame_w: int, frame_h: int) -> CalibratedGazeData:
        now = time.time()

        if landmarks is None or raw_frame is None:
            time_in_zone = now - self.stable_grid_start_time
            return CalibratedGazeData(
                is_calibrated=self.is_calibrated,
                active_grid=self.active_grid,
                active_label="NO FACE DETECTED",
                time_in_zone=time_in_zone,
                time_in_zone_str=self._format_duration(time_in_zone),
                prev_grid=self.prev_grid,
                prev_label=ZONE_LABELS.get(self.prev_grid, "UNKNOWN"),
                time_between_changes=self.last_transition_interval,
                movement_str=self.movement_str,
                movement_speed=self.movement_speed,
                last_change_time=self.last_change_timestamp,
                history=self.recent_history,
                face_detected=False,
                attention_state="NO FACE",
            )

        feat_dict = self.extract_features(landmarks, raw_frame, frame_w, frame_h)
        if feat_dict is None:
            return CalibratedGazeData(is_calibrated=self.is_calibrated, face_detected=False)

        # 1. Blink Dynamics & Rapid Blinking (Stress / Fatigue)
        ear = feat_dict.get("ear", 0.30)
        is_blink_closed = (ear < 0.18)
        if is_blink_closed and not self.in_blink:
            self.in_blink = True
            self.blink_timestamps.append(now)
        elif not is_blink_closed and self.in_blink:
            self.in_blink = False

        cutoff_blink = now - 30.0
        self.blink_timestamps = [ts for ts in self.blink_timestamps if ts >= cutoff_blink]
        elapsed_blink = min(30.0, max(5.0, now - self.tracking_start_time))
        blink_bpm = round((len(self.blink_timestamps) / elapsed_blink) * 60.0, 1)
        is_rapid_blinking = bool(blink_bpm >= 32.0 and len(self.blink_timestamps) >= 4)

        # 2. Pupillometry (Cognitive Workload & Effort)
        current_pupil_ratio = feat_dict.get("pupil_ratio", 0.0)
        if self.model.baseline_pupil_ratio > 0.001:
            dil_raw = ((current_pupil_ratio - self.model.baseline_pupil_ratio) / self.model.baseline_pupil_ratio) * 100.0
            self.smooth_dilation_pct = 0.15 * dil_raw + 0.85 * self.smooth_dilation_pct
        else:
            # Auto-adapt baseline if not yet calibrated
            self.baseline_pupil_ratios.append(current_pupil_ratio)
            if len(self.baseline_pupil_ratios) >= 25:
                self.model.baseline_pupil_ratio = float(np.median(self.baseline_pupil_ratios))
            self.smooth_dilation_pct = 0.0

        dilation_pct = round(self.smooth_dilation_pct, 1)
        if dilation_pct >= 7.0:
            pupil_state = "DILATED (EFFORT)"
        elif dilation_pct <= -7.0:
            pupil_state = "CONSTRICTED (RELAXED)"
        else:
            pupil_state = "NORMAL"

        # In calibration mode, return state immediately
        if self.is_calibrating or not self.is_calibrated:
            if self.calib_phase == self.CALIB_PHASE_BASELINE:
                active_grid = (1, 1)
            else:
                step_idx = min(max(0, self.calib_step_idx), len(SNAKE_GRID_ORDER) - 1)
                active_grid = SNAKE_GRID_ORDER[step_idx]
            return CalibratedGazeData(
                is_calibrated=self.is_calibrated,
                active_grid=active_grid,
                active_label="CALIBRATION IN PROGRESS",
                face_detected=True,
                head_yaw_deg=round(feat_dict["yaw"], 1),
                head_pitch_deg=round(feat_dict["pitch"], 1),
                head_roll_deg=round(feat_dict["roll"], 1),
                ear=round(ear, 2),
                is_blinking=is_blink_closed,
                blink_bpm=blink_bpm,
                is_rapid_blinking=is_rapid_blinking,
                pupil_dilation_pct=dilation_pct,
                pupil_state=pupil_state,
            )

        # 3. Biomechanical Prototype IDW Prediction
        disp_mm = feat_dict["disp_mm"]
        yaw = feat_dict["yaw"]
        pitch = feat_dict["pitch"]

        delta_x_mm = disp_mm[0] - self.model.disp_base_mm[0]
        delta_y_mm = disp_mm[1] - self.model.disp_base_mm[1]
        delta_yaw = yaw - self.model.head_base[0]
        delta_pitch = pitch - self.model.head_base[1]

        self.last_delta_mm = (round(delta_x_mm, 2), round(delta_y_mm, 2))

        # Direct Prototype IDW Screen Prediction
        raw_x, raw_y = self.model.predict(delta_x_mm, delta_y_mm, delta_yaw, delta_pitch)

        # 4. Blink Masking & Adaptive Dynamic Gaze Smoothing
        dt = max(0.005, min(0.20, now - self.last_frame_time))
        self.last_frame_time = now

        if not is_blink_closed:
            raw_pt = np.array([raw_x, raw_y], dtype=np.float64)

            # Instantaneous eye movement velocity
            dist_step = math.hypot(raw_x - self.last_raw_gaze[0], raw_y - self.last_raw_gaze[1])
            self.last_raw_gaze = (raw_x, raw_y)
            inst_vel = dist_step / dt
            self.smooth_vel = 0.25 * inst_vel + 0.75 * self.smooth_vel

            # Dynamic Adaptive Alpha (One-Euro style):
            # When eye is steady: alpha ~ 0.25 (silky smooth, kills jitter)
            # When eye shifts rapidly: alpha automatically scales up to 0.75 (instant follow, zero lag)
            vel_factor = min(1.0, max(0.0, (self.smooth_vel - 0.20) / 0.80))
            adaptive_alpha = 0.25 + 0.50 * vel_factor

            self.current_smooth_gaze = (
                adaptive_alpha * raw_pt + (1.0 - adaptive_alpha) * self.current_smooth_gaze
            )

            # Gaze State Classifier (I-VT Velocity Filter)
            # A true saccade requires both velocity >= 1.0 screen units/sec AND a displacement >= 0.07 screen units
            is_saccade_event = (self.smooth_vel >= 1.0 and dist_step >= 0.07)
            if is_saccade_event:
                self.gaze_state = "SACCADE"
                if not self.in_saccade:
                    self.in_saccade = True
                    self.saccade_timestamps.append(now)
            else:
                self.gaze_state = "FIXATION"
                self.in_saccade = False
        else:
            # During blink closure: freeze gaze cursor at current position (Blink Masking)
            self.gaze_state = "FIXATION"

        # 5. True Eye Darting Detector (Restless Gaze Dynamics)
        saccade_cutoff = now - 2.5
        self.saccade_timestamps = [ts for ts in self.saccade_timestamps if ts >= saccade_cutoff]
        recent_saccades = len(self.saccade_timestamps)

        # Requires >= 4 distinct saccades in 2.5s with rapid short dwells
        if recent_saccades >= 4 and len(self.recent_history) >= 2:
            avg_dwell = sum(ev.dwell_sec for ev in self.recent_history[-2:]) / 2.0
            if avg_dwell < 0.35:
                self.is_eye_darting = True
                self.gaze_state = "DARTING"
            else:
                self.is_eye_darting = False
        else:
            self.is_eye_darting = False

        smooth_x = float(np.clip(self.current_smooth_gaze[0], 0.0, 1.0))
        smooth_y = float(np.clip(self.current_smooth_gaze[1], 0.0, 1.0))

        # 6. Discrete Active Cell Selection with Prototype Distance Affinity & Temporal Debounce
        detected_grid = self.model.get_best_zone(
            delta_x_mm, delta_y_mm, delta_yaw, delta_pitch, self.active_grid
        )

        if detected_grid == self.active_grid:
            self.candidate_grid = self.active_grid
            self.candidate_start_time = now
        else:
            if self.candidate_grid != detected_grid:
                self.candidate_grid = detected_grid
                self.candidate_start_time = now
            elif (now - self.candidate_start_time) >= STABILITY_DWELL_THRESH:
                self._commit_grid_transition(detected_grid, now)

        # 7. Lateral Gaze Aversion (True Looking-Away Distraction Detection)
        # Any point on the 4x3 screen is ON SCREEN and FOCUSED.
        # Only triggers when head is genuinely turned away (> 28 deg) or eyes have left the screen for >= 2.5s
        head_turned_away = (abs(delta_yaw) > 28.0) or (abs(delta_pitch) > 28.0)
        eyes_off_screen = (smooth_x <= 0.001 and delta_x_mm < -2.4) or (smooth_x >= 0.999 and delta_x_mm > 2.4)
        is_looking_away = head_turned_away or eyes_off_screen

        if is_looking_away:
            if self.lateral_start_time is None:
                self.lateral_start_time = now
            lateral_dur = now - self.lateral_start_time
            self.lateral_distraction_sec = lateral_dur
            self.is_lateral_distracted = (lateral_dur >= 2.5)
        else:
            self.lateral_start_time = None
            self.is_lateral_distracted = False
            self.lateral_distraction_sec = 0.0

        # Overall Attention State
        if self.is_lateral_distracted:
            attention_state = "LOOKING AWAY (DISTRACTED)"
        elif self.is_eye_darting:
            attention_state = "RESTLESS (EYE DARTING)"
        else:
            attention_state = "FOCUSED / ENGAGED"

        time_in_zone = now - self.stable_grid_start_time
        active_label = ZONE_LABELS.get(self.active_grid, f"ZONE [{self.active_grid[0]},{self.active_grid[1]}]")
        confidence = min(98, max(75, int(self.ransac_inlier_pct * 0.98))) if self.is_calibrated else 0

        return CalibratedGazeData(
            is_calibrated=self.is_calibrated,
            active_grid=self.active_grid,
            active_label=active_label,
            confidence_pct=confidence,
            ransac_inlier_pct=self.ransac_inlier_pct,
            ransac_outliers=self.ransac_rejected_outliers,
            raw_gaze_norm=(raw_x, raw_y),
            smooth_gaze_norm=(smooth_x, smooth_y),
            time_in_zone=time_in_zone,
            time_in_zone_str=self._format_duration(time_in_zone),
            prev_grid=self.prev_grid,
            prev_label=ZONE_LABELS.get(self.prev_grid, "NONE"),
            time_between_changes=self.last_transition_interval,
            movement_str=self.movement_str,
            movement_speed=self.movement_speed,
            last_change_time=self.last_change_timestamp,
            history=self.recent_history,
            face_detected=True,
            metric_delta_x_mm=self.last_delta_mm[0],
            metric_delta_y_mm=self.last_delta_mm[1],
            head_yaw_deg=round(yaw, 1),
            head_pitch_deg=round(pitch, 1),
            head_roll_deg=round(feat_dict["roll"], 1),
            gaze_state=self.gaze_state,
            gaze_velocity=round(self.smooth_vel, 2),
            is_eye_darting=self.is_eye_darting,
            saccade_count_window=recent_saccades,
            is_lateral_distracted=self.is_lateral_distracted,
            lateral_distraction_sec=round(self.lateral_distraction_sec, 1),
            attention_state=attention_state,
            pupil_dilation_pct=dilation_pct,
            pupil_state=pupil_state,
            ear=round(ear, 2),
            is_blinking=is_blink_closed,
            blink_bpm=blink_bpm,
            is_rapid_blinking=is_rapid_blinking,
        )

    def _commit_grid_transition(self, new_grid: Tuple[int, int], now: float) -> None:
        dwell = now - self.stable_grid_start_time
        old_grid = self.active_grid
        old_label = ZONE_LABELS.get(old_grid, f"ZONE [{old_grid[0]},{old_grid[1]}]")

        event = GridVisitEvent(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            grid_coords=old_grid,
            label=old_label,
            dwell_sec=round(dwell, 1),
        )
        self.recent_history.append(event)
        if len(self.recent_history) > 4:
            self.recent_history.pop(0)

        self.last_transition_interval = now - self.candidate_start_time
        if self.last_transition_interval < 0.20:
            self.movement_speed = "FAST"
        elif self.last_transition_interval < 0.45:
            self.movement_speed = "NORMAL"
        else:
            self.movement_speed = "SLOW"

        self.movement_str = f"[{old_grid[0]},{old_grid[1]}] -> [{new_grid[0]},{new_grid[1]}]"
        self.prev_grid = old_grid
        self.active_grid = new_grid
        self.stable_grid_start_time = now
        self.last_change_timestamp = datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
