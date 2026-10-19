import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional, Tuple

# ==============================================================================
# CONFIGURABLE PARAMETERS & THRESHOLDS
# ==============================================================================
# Landmark Exponential Moving Average smoothing factor
EMA_ALPHA = 0.45

# Movement velocity thresholds (normalized screen units per second)
STILL_NOISE_FLOOR = 0.016      # Below this is resting breathing / camera sensor noise
MOVE_ACTIVE_VELOCITY = 0.025   # At or above this triggers active movement
MOVE_EPISODE_COOLDOWN = 1.0    # Seconds between distinct movement episodes for frequency count

# Posture deviation thresholds (relative to student's calibrated resting baseline)
SHOULDER_TILT_DEVIATION_DEG = 5.5  # Degrees deviation from baseline to trigger tilt
HEAD_LATERAL_DEVIATION = 0.12      # Ratio of shoulder width to trigger head tilt
HEAD_FORWARD_DEVIATION = 0.15      # Vertical drop in head height indicating forward head
TORSO_LEAN_DEVIATION = 0.065       # Normalized lateral shift from baseline

# Repositioning threshold
REPOSITION_SHIFT_THRESH = 0.038    # Shift required to count as a major body repositioning

# Rolling history settings
FREQ_WINDOW_SECONDS = 60.0         # Rolling 60s window to calculate movement frequency
TREND_BUFFER_SIZE = 28             # Number of samples in the activity sparkline
TREND_SAMPLE_INTERVAL = 1.5        # Seconds per sparkline sample


@dataclass
class PostureData:
    head_state: str = "CENTERED"
    shoulder_state: str = "LEVEL"
    torso_state: str = "CENTERED"
    current_state: str = "STABLE"
    time_in_state_str: str = "00:00"
    movement_magnitude: str = "NONE"
    movement_frequency: str = "0/min"
    movement_speed: str = "STILL"
    movement_direction: str = "STILL"
    repositioning_count: int = 0
    last_change_time: str = "--:--:--"
    time_between_changes_str: str = "--:--"
    activity_history: List[float] = field(default_factory=list)
    is_calibrated: bool = False
    calibrated_head_pos: Optional[Tuple[float, float]] = None
    current_head_pos: Optional[Tuple[float, float]] = None
    head_box_size: float = 0.20
    is_head_in_neutral_box: bool = True
    person_detected: bool = False


class PostureDetector:
    """
    Robust geometric posture analyzer and movement dynamics engine.
    Uses adaptive baseline calibration, instant recalibration, and
    calibrated velocity-based movement episode detection.
    """

    def __init__(self):
        now = time.time()
        self.session_start = now

        # Smoothed landmarks dictionary: {id: (x, y, z)}
        self.smooth_points: dict = {}
        self.prev_body_anchor: Optional[Tuple[float, float]] = None
        self.last_update_time: float = now

        # Calibrated baselines (normal seated study posture)
        self.is_calibrated: bool = False
        self.calibration_samples: int = 0
        self.base_shoulder_angle: float = 0.0
        self.base_shoulder_width: float = 0.35
        self.base_head_lat_offset: float = 0.0
        self.base_head_vert_height: float = 0.25
        self.base_mid_shoulder_x: float = 0.50
        self.base_head_pos: Tuple[float, float] = (0.50, 0.35)

        # Posture state machine
        self.current_posture_state: str = "STABLE"
        self.state_start_time: float = now
        self.last_reposition_time: float = now
        self.last_change_timestamp: str = datetime.now().strftime("%H:%M:%S")
        self.repositioning_count: int = 0
        self.last_time_between_changes: float = 0.0

        # Movement episode tracking
        self.is_moving: bool = False
        self.last_episode_trigger_time: float = 0.0
        self.current_magnitude: str = "NONE"
        self.current_speed: str = "STILL"
        self.current_direction: str = "STILL"
        self.move_events_history: Deque[float] = deque()  # Timestamps of movement episodes

        # Rolling activity trend buffer
        self.activity_trend: Deque[float] = deque([0.0] * TREND_BUFFER_SIZE, maxlen=TREND_BUFFER_SIZE)
        self.last_trend_sample_time: float = now
        self.trend_accumulator: float = 0.0
        self.trend_sample_count: int = 0

    def calibrate(self) -> None:
        """Forces an instant re-calibration to the current sitting position."""
        self.is_calibrated = False
        self.calibration_samples = 0

    def update(self, pose_landmarks, frame_width: int, frame_height: int) -> PostureData:
        current_time = time.time()
        dt = max(0.016, min(0.10, current_time - self.last_update_time))

        if pose_landmarks is None:
            time_in_state = current_time - self.state_start_time
            return PostureData(
                head_state="NO PERSON",
                shoulder_state="NO PERSON",
                torso_state="NO PERSON",
                current_state="NO PERSON",
                time_in_state_str=self._format_duration(time_in_state),
                movement_magnitude="NONE",
                movement_frequency=f"{self._calculate_frequency(current_time)}/min",
                movement_speed="STILL",
                movement_direction="STILL",
                repositioning_count=self.repositioning_count,
                last_change_time=self.last_change_timestamp,
                time_between_changes_str=self._format_duration(self.last_time_between_changes),
                activity_history=list(self.activity_trend),
                is_calibrated=self.is_calibrated,
                person_detected=False,
            )

        # 1. Exponential Moving Average smoothing
        self._smooth_landmarks(pose_landmarks)

        p11 = self.smooth_points.get(11)
        p12 = self.smooth_points.get(12)
        if not p11 or not p12:
            return PostureData(person_detected=False)

        # Screen-oriented shoulders (leftmost on screen vs rightmost on screen)
        sh_l = p11 if p11[0] < p12[0] else p12
        sh_r = p12 if p11[0] < p12[0] else p11

        mid_sh = ((sh_l[0] + sh_r[0]) / 2.0, (sh_l[1] + sh_r[1]) / 2.0)
        sh_width = math.dist((sh_l[0], sh_l[1]), (sh_r[0], sh_r[1]))

        # Calculate current shoulder angle
        dx_px = (sh_r[0] - sh_l[0]) * frame_width
        dy_px = (sh_r[1] - sh_l[1]) * frame_height
        raw_shoulder_angle = math.degrees(math.atan2(dy_px, dx_px))

        # Head center position
        p0 = self.smooth_points.get(0)
        p7 = self.smooth_points.get(7)
        p8 = self.smooth_points.get(8)

        if p7 and p8:
            head_x = (p7[0] + p8[0]) / 2.0
            head_y = (p7[1] + p8[1]) / 2.0
            ear_dist = math.dist((p7[0], p7[1]), (p8[0], p8[1]))
            head_box_size = max(0.12, ear_dist * 1.5)
        elif p0:
            head_x, head_y = p0[0], p0[1]
            head_box_size = max(0.12, sh_width * 0.5)
        else:
            head_x, head_y = mid_sh[0], max(0.05, mid_sh[1] - 0.25)
            head_box_size = 0.15

        head_lat_offset = (head_x - mid_sh[0]) / max(0.01, sh_width)
        head_vert_height = mid_sh[1] - head_y

        # 2. Adaptive Calibration (first 25 frames establish normal baseline)
        if not self.is_calibrated:
            self.calibration_samples += 1
            w = 1.0 / self.calibration_samples
            self.base_shoulder_angle = (1 - w) * self.base_shoulder_angle + w * raw_shoulder_angle
            self.base_shoulder_width = (1 - w) * self.base_shoulder_width + w * sh_width
            self.base_head_lat_offset = (1 - w) * self.base_head_lat_offset + w * head_lat_offset
            self.base_head_vert_height = (1 - w) * self.base_head_vert_height + w * head_vert_height
            self.base_mid_shoulder_x = (1 - w) * self.base_mid_shoulder_x + w * mid_sh[0]
            self.base_head_pos = (
                (1 - w) * self.base_head_pos[0] + w * head_x,
                (1 - w) * self.base_head_pos[1] + w * head_y,
            )

            if self.calibration_samples >= 25:
                self.is_calibrated = True

        # 3. Geometric Posture Evaluation
        shoulder_state = self._eval_shoulders(raw_shoulder_angle)
        head_state = self._eval_head(head_lat_offset, head_vert_height)
        torso_state = self._eval_torso(mid_sh[0], sh_width)

        # Check if head is within the neutral calibrated square
        head_shift_x = abs(head_x - self.base_head_pos[0])
        head_shift_y = abs(head_y - self.base_head_pos[1])
        neutral_tolerance = head_box_size * 0.35
        is_head_neutral = (head_shift_x <= neutral_tolerance) and (head_shift_y <= neutral_tolerance)

        # 4. Movement Dynamics & Velocity Tracking
        velocity, move_vec = self._compute_velocity(mid_sh, head_x, head_y, dt)
        self._process_movement(velocity, move_vec, mid_sh, current_time)

        # 5. Activity Trend Tracking (Sparkline)
        self._update_activity_trend(velocity, current_time)

        # 6. Dwell time in current state
        time_in_state = current_time - self.state_start_time
        self.last_update_time = current_time

        return PostureData(
            head_state=head_state,
            shoulder_state=shoulder_state,
            torso_state=torso_state,
            current_state=self.current_posture_state,
            time_in_state_str=self._format_duration(time_in_state),
            movement_magnitude=self.current_magnitude,
            movement_frequency=f"{self._calculate_frequency(current_time)}/min",
            movement_speed=self.current_speed,
            movement_direction=self.current_direction,
            repositioning_count=self.repositioning_count,
            last_change_time=self.last_change_timestamp,
            time_between_changes_str=self._format_duration(self.last_time_between_changes),
            activity_history=list(self.activity_trend),
            is_calibrated=self.is_calibrated,
            calibrated_head_pos=self.base_head_pos,
            current_head_pos=(head_x, head_y),
            head_box_size=head_box_size,
            is_head_in_neutral_box=is_head_neutral,
            person_detected=True,
        )

    def _smooth_landmarks(self, pose_landmarks) -> None:
        """Applies Exponential Moving Average smoothing across all landmarks."""
        for i, lm in enumerate(pose_landmarks.landmark):
            curr = (lm.x, lm.y, lm.z)
            if i in self.smooth_points:
                prev = self.smooth_points[i]
                self.smooth_points[i] = (
                    EMA_ALPHA * curr[0] + (1 - EMA_ALPHA) * prev[0],
                    EMA_ALPHA * curr[1] + (1 - EMA_ALPHA) * prev[1],
                    EMA_ALPHA * curr[2] + (1 - EMA_ALPHA) * prev[2],
                )
            else:
                self.smooth_points[i] = curr

    def _eval_shoulders(self, current_angle: float) -> str:
        dev = current_angle - self.base_shoulder_angle
        if abs(dev) <= SHOULDER_TILT_DEVIATION_DEG:
            return "LEVEL"
        if dev > SHOULDER_TILT_DEVIATION_DEG:
            return "TILT RIGHT"
        return "TILT LEFT"

    def _eval_head(self, current_lat_offset: float, current_vert_height: float) -> str:
        lat_dev = current_lat_offset - self.base_head_lat_offset

        if self.base_head_vert_height > 0.05:
            if current_vert_height < self.base_head_vert_height * (1.0 - HEAD_FORWARD_DEVIATION):
                return "FORWARD HEAD"

        if lat_dev < -HEAD_LATERAL_DEVIATION:
            return "TILT LEFT"
        if lat_dev > HEAD_LATERAL_DEVIATION:
            return "TILT RIGHT"
        return "CENTERED"

    def _eval_torso(self, mid_sh_x: float, sh_width: float) -> str:
        lat_dev = mid_sh_x - self.base_mid_shoulder_x

        if lat_dev < -TORSO_LEAN_DEVIATION:
            return "LEANING LEFT"
        if lat_dev > TORSO_LEAN_DEVIATION:
            return "LEANING RIGHT"

        if self.base_shoulder_width > 0.05:
            depth_ratio = sh_width / self.base_shoulder_width
            if depth_ratio > 1.15:
                return "LEANING FORWARD"
            if depth_ratio < 0.86:
                return "LEANING BACK"

        return "CENTERED"

    def _compute_velocity(self, mid_sh, head_x, head_y, dt: float) -> Tuple[float, Tuple[float, float]]:
        """Calculates instantaneous velocity in normalized units per second."""
        curr_anchor = (mid_sh[0] * 0.6 + head_x * 0.4, mid_sh[1] * 0.6 + head_y * 0.4)

        if self.prev_body_anchor is None:
            self.prev_body_anchor = curr_anchor
            return 0.0, (0.0, 0.0)

        dx = curr_anchor[0] - self.prev_body_anchor[0]
        dy = curr_anchor[1] - self.prev_body_anchor[1]
        disp = math.hypot(dx, dy)
        vel = disp / dt

        self.prev_body_anchor = curr_anchor
        return vel, (dx, dy)

    def _process_movement(self, vel: float, move_vec: Tuple[float, float], mid_sh, current_time: float) -> None:
        """Groups continuous movements into episodes and updates frequency."""
        # Clean rolling frequency cleanup
        cutoff = current_time - FREQ_WINDOW_SECONDS
        while self.move_events_history and self.move_events_history[0] < cutoff:
            self.move_events_history.popleft()

        if vel >= MOVE_ACTIVE_VELOCITY:
            # Active movement episode
            if not self.is_moving:
                self.is_moving = True
                self.current_posture_state = "SHIFTING"

                # Check cooldown to record a distinct movement event
                if current_time - self.last_episode_trigger_time >= MOVE_EPISODE_COOLDOWN:
                    self.move_events_history.append(current_time)
                    self.last_episode_trigger_time = current_time

            # Update live movement attributes
            self.current_magnitude = self._classify_magnitude(vel)
            self.current_speed = self._classify_speed(vel)
            self.current_direction = self._classify_direction(move_vec)

        elif vel < STILL_NOISE_FLOOR:
            # Calm / Stationary state
            if self.is_moving:
                # Movement ended; check if user settled into a new position
                shift = abs(mid_sh[0] - self.base_mid_shoulder_x)
                if shift >= REPOSITION_SHIFT_THRESH:
                    self._commit_repositioning(mid_sh[0], current_time)

                self.is_moving = False
                self.current_posture_state = "STABLE"
                self.current_magnitude = "NONE"
                self.current_speed = "STILL"
                self.current_direction = "STILL"

    def _commit_repositioning(self, new_mid_x: float, current_time: float) -> None:
        self.repositioning_count += 1
        self.last_time_between_changes = current_time - self.last_reposition_time
        self.last_reposition_time = current_time
        self.last_change_timestamp = datetime.now().strftime("%H:%M:%S")
        self.state_start_time = current_time
        self.base_mid_shoulder_x = new_mid_x

    def _classify_magnitude(self, vel: float) -> str:
        if vel < 0.055:
            return "SLIGHT"
        if vel < 0.130:
            return "MODERATE"
        return "LARGE"

    def _classify_speed(self, vel: float) -> str:
        if vel < 0.055:
            return "SLOW"
        if vel < 0.130:
            return "NORMAL"
        return "FAST"

    def _classify_direction(self, vec: Tuple[float, float]) -> str:
        dx, dy = vec
        if math.hypot(dx, dy) < 0.001:
            return "STILL"
        if abs(dx) > abs(dy) * 1.3:
            return "SHIFT RIGHT" if dx > 0 else "SHIFT LEFT"
        if abs(dy) > abs(dx) * 1.3:
            return "DOWN" if dy > 0 else "UP"
        return "MOVING"

    def _calculate_frequency(self, current_time: float) -> int:
        cutoff = current_time - FREQ_WINDOW_SECONDS
        while self.move_events_history and self.move_events_history[0] < cutoff:
            self.move_events_history.popleft()
        return len(self.move_events_history)

    def _update_activity_trend(self, vel: float, current_time: float) -> None:
        self.trend_accumulator += vel
        self.trend_sample_count += 1

        if current_time - self.last_trend_sample_time >= TREND_SAMPLE_INTERVAL:
            avg_energy = self.trend_accumulator / max(1, self.trend_sample_count)
            norm_energy = min(1.0, avg_energy / 0.10)
            self.activity_trend.append(norm_energy)
            self.trend_accumulator = 0.0
            self.trend_sample_count = 0
            self.last_trend_sample_time = current_time

    @staticmethod
    def _format_duration(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
