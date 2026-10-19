"""
Typing Dynamics Analysis Engine
Focus Detection Model for Students - Behavioral Sensing Module

Derives typing frequency, inter-key intervals (IKI), bursts, pauses,
speed, cadence consistency, error corrections, and persistence metrics
from raw keystroke event timings without storing actual typed content.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional, Tuple


# ==============================================================================
# CONFIGURABLE PARAMETERS & THRESHOLDS
# ==============================================================================
BURST_PAUSE_THRESHOLD = 1.4         # IKI greater than this ends a burst and starts a pause (seconds)
INACTIVITY_THRESHOLD = 2.5          # Idle time to classify state as PAUSED / INACTIVE (seconds)
INTERMITTENT_IKI_THRESHOLD = 0.75   # IKI threshold between ACTIVE and INTERMITTENT typing (seconds)
ROLLING_WINDOW_SEC = 60.0           # Rolling window for frequency, WPM, and pause rate (seconds)
POST_CORRECTION_WINDOW_SEC = 5.0    # Time window after deletion to register effort resumption (seconds)

# Temporal Timeline Settings
TIMELINE_SAMPLE_INTERVAL = 1.0      # Seconds per timeline graph sample
TIMELINE_BUFFER_SIZE = 45           # Number of samples visible on the timeline canvas

# Revision Cycle Thresholds
REVISION_MIN_PRE_KEYS = 3           # Minimum content keys before deletion
REVISION_MIN_DEL_KEYS = 2           # Minimum deletions in a run
REVISION_MIN_POST_KEYS = 3          # Minimum content keys after deletion to complete cycle


@dataclass
class GraphPoint:
    """Represents a discrete temporal sample on the typing graph."""
    wpm: float = 0.0                # Instantaneous WPM rate during this interval
    intensity: float = 0.0          # Normalized 0.0 to 1.0 (relative to 80 WPM)
    has_backspace: bool = False     # True if backspace/delete was used during this sample
    backspace_count: int = 0        # Number of backspaces in this interval


@dataclass
class TypingDynamicsData:
    """Live snapshot of typing behavioral dynamics for human-readable display."""
    # 1. Activity State
    activity_state: str = "INACTIVE"        # ACTIVE TYPING | INTERMITTENT | PAUSED | INACTIVE
    time_in_state_str: str = "00:00"
    is_in_burst: bool = False
    current_burst_keys: int = 0
    total_bursts_count: int = 0

    # 2. Speed & Frequency
    gross_wpm: float = 0.0                  # Rolling 60-second WPM
    session_wpm: float = 0.0                # Session-wide average WPM
    keystrokes_per_min: int = 0             # Rolling 60-second KPM
    total_keystrokes: int = 0

    # 3. Inter-Key Intervals & Consistency
    current_iki_ms: int = 0                 # Milliseconds since previous key
    mean_iki_ms: int = 0                    # Mean IKI over recent typing window
    consistency_label: str = "STEADY"       # STEADY | MODERATE | VARIABLE
    consistency_cv: float = 0.0             # Coefficient of variation (std_dev / mean)

    # 4. Pauses
    current_pause_sec: float = 0.0          # Active idle time
    total_pauses_count: int = 0
    pause_frequency_str: str = "0/min"
    avg_pause_sec: float = 0.0

    # 5. Error Correction & Persistence
    total_deletions: int = 0
    deletion_rate_pct: float = 0.0          # (total_deletions / total_keystrokes) * 100
    deletion_bursts_count: int = 0
    max_deletion_burst: int = 0
    avg_time_to_correction_sec: float = 0.0 # Latency between content typing and correction
    revision_cycles_count: int = 0          # Cycles of Type -> Delete -> Rewrite
    post_correction_resumptions: int = 0    # Continued typing after corrections
    resumption_rate_pct: float = 0.0        # Resumption efficiency percentage

    # 6. Session Timings
    session_duration_str: str = "00:00"
    last_event_time_str: str = "--:--:--"

    # 7. Timeline Graph History: 45 samples of GraphPoint(wpm, intensity, has_backspace, backspace_count)
    timeline_history: List[GraphPoint] = field(default_factory=list)


class TypingDynamicsAnalyzer:
    """
    Core behavioral dynamics engine for typing analysis.
    Processes keystroke event timings and behavioral metadata.
    Does NOT store or log the characters or text typed.
    """

    def __init__(self):
        now = time.time()
        self.session_start = now
        self.last_press_time: Optional[float] = None
        self.last_content_press_time: Optional[float] = None
        self.last_update_time = now

        # State Machine
        self.current_state = "INACTIVE"
        self.state_start_time = now
        self.last_event_timestamp_str = "--:--:--"

        # Keystroke Records (Timestamps only for privacy)
        self.total_keystrokes = 0
        self.total_deletions = 0
        self.rolling_keystrokes: Deque[float] = deque()  # Timestamps of presses within 60s

        # Inter-Key Intervals (seconds)
        self.recent_ikis: Deque[float] = deque(maxlen=60)
        self.all_ikis: List[float] = []

        # Typing Bursts
        self.is_in_burst = False
        self.current_burst_keys = 0
        self.current_burst_start = 0.0
        self.total_bursts_count = 0

        # Pauses
        self.total_pauses_count = 0
        self.pause_durations: List[float] = []
        self.rolling_pauses: Deque[float] = deque()      # Timestamps of pauses within 60s
        self.is_paused_idle = False

        # Error Corrections & Deletion Bursts
        self.current_del_run = 0
        self.del_run_start_time = 0.0
        self.time_to_first_corr_list: List[float] = []
        self.deletion_bursts_count = 0
        self.max_deletion_burst = 0

        # Persistence & Revision Tracking
        self.pending_resumption = False
        self.pending_resumption_time = 0.0
        self.post_correction_resumptions = 0

        # State Tracker for Revision Cycles (Type -> Delete -> Rewrite)
        # States: 0 = Idle/Initial, 1 = Content typed (>=3), 2 = Deletion run completed (>=2)
        self.revision_state = 0
        self.pre_del_content_count = 0
        self.post_del_content_count = 0
        self.revision_cycles_count = 0

        # Thread Safety Lock for background listeners
        self._lock = threading.Lock()

        # Temporal Timeline Buffer
        self.timeline_data: Deque[GraphPoint] = deque(
            [GraphPoint() for _ in range(TIMELINE_BUFFER_SIZE)], maxlen=TIMELINE_BUFFER_SIZE
        )
        self.last_timeline_sample_time = now
        self.accumulated_keys_in_interval = 0
        self.accumulated_deletions_in_interval = 0
        self.current_interval_marker = "NONE"

    def record_key_event(self, event_type: str, key_category: str) -> None:
        """
        Record a keyboard event with current timestamp (Thread-safe).
        
        Args:
            event_type: 'PRESS' or 'RELEASE'
            key_category: 'CHAR', 'SPACE', 'ENTER', 'BACKSPACE', 'DELETE', 'NAV', 'MOD'
        """
        now = time.time()
        with self._lock:
            self.last_event_timestamp_str = datetime.now().strftime("%H:%M:%S")

            if event_type == "PRESS":
                self._handle_key_press(key_category, now)
            elif event_type == "RELEASE":
                self._handle_key_release(key_category, now)

    def _handle_key_press(self, key_category: str, now: float) -> None:
        # Ignore modifier keys (Shift, Ctrl, Alt) for cadence and frequency counts
        if key_category == "MOD":
            return

        self.total_keystrokes += 1
        self.rolling_keystrokes.append(now)
        self.accumulated_keys_in_interval += 1

        is_deletion = (key_category in ("BACKSPACE", "DELETE"))
        is_content = (key_category in ("CHAR", "SPACE", "ENTER"))

        # 1. Inter-Key Interval (IKI)
        if self.last_press_time is not None:
            iki = now - self.last_press_time
            if iki > 0.005:  # Filter out duplicate zero-delay OS key repeats
                self.recent_ikis.append(iki)
                self.all_ikis.append(iki)

                if self.is_paused_idle:
                    # Resuming after an inactivity pause
                    self.is_paused_idle = False
                    self.pause_durations.append(iki)
                    if self.is_in_burst:
                        self._finish_current_burst()
                    self._start_new_burst(now)
                elif iki >= BURST_PAUSE_THRESHOLD:
                    self._commit_pause(iki, now)
                    if self.is_in_burst:
                        self._finish_current_burst()
                    self._start_new_burst(now)
                else:
                    # Sustained typing inside burst
                    if not self.is_in_burst:
                        self._start_new_burst(now)
                    self.current_burst_keys += 1
        else:
            self._start_new_burst(now)

        self.last_press_time = now

        # 2. Deletion and Error Correction Tracking
        if is_deletion:
            self.total_deletions += 1
            self.accumulated_deletions_in_interval += 1
            if self.current_del_run == 0:
                # Beginning of a deletion burst
                self.del_run_start_time = now
                if self.last_content_press_time is not None:
                    time_to_corr = now - self.last_content_press_time
                    self.time_to_first_corr_list.append(time_to_corr)
            self.current_del_run += 1
            self.max_deletion_burst = max(self.max_deletion_burst, self.current_del_run)
            self.current_interval_marker = "DELETION"

            # Revision Cycle tracking: step 1 -> 2
            if self.revision_state == 1 and self.current_del_run >= REVISION_MIN_DEL_KEYS:
                self.revision_state = 2
                self.post_del_content_count = 0

        elif is_content:
            # End of an active deletion run
            if self.current_del_run > 0:
                self._finish_deletion_run(now)

            self.last_content_press_time = now

            # Check persistence: did user resume typing after an error within threshold?
            if self.pending_resumption:
                if (now - self.pending_resumption_time) <= POST_CORRECTION_WINDOW_SEC:
                    self.post_correction_resumptions += 1
                self.pending_resumption = False

            # Revision Cycle tracking: step 2 -> 3 (Rewrite completed)
            if self.revision_state == 2:
                self.post_del_content_count += 1
                if self.post_del_content_count >= REVISION_MIN_POST_KEYS:
                    self.revision_cycles_count += 1
                    self.current_interval_marker = "REVISION"
                    self.revision_state = 1  # Reset to drafting state
                    self.pre_del_content_count = self.post_del_content_count
            else:
                self.pre_del_content_count += 1
                if self.pre_del_content_count >= REVISION_MIN_PRE_KEYS:
                    self.revision_state = 1

        # 3. Dynamic Activity State Update
        self._evaluate_activity_state(now)

    def _handle_key_release(self, key_category: str, now: float) -> None:
        pass  # Reserved for key-hold dwell time analysis if needed

    def _start_new_burst(self, now: float) -> None:
        self.is_in_burst = True
        self.current_burst_keys = 1
        self.current_burst_start = now
        self.total_bursts_count += 1
        if self.current_interval_marker == "NONE":
            self.current_interval_marker = "BURST"

    def _finish_current_burst(self) -> None:
        self.is_in_burst = False
        self.current_burst_keys = 0

    def _commit_pause(self, duration: float, now: float) -> None:
        self.total_pauses_count += 1
        self.pause_durations.append(duration)
        self.rolling_pauses.append(now)
        if self.current_interval_marker == "NONE":
            self.current_interval_marker = "PAUSE"

    def _finish_deletion_run(self, now: float) -> None:
        self.deletion_bursts_count += 1
        self.current_del_run = 0
        self.pending_resumption = True
        self.pending_resumption_time = now

    def _evaluate_activity_state(self, now: float) -> None:
        if self.last_press_time is None:
            new_state = "INACTIVE"
        else:
            idle = now - self.last_press_time
            if idle >= INACTIVITY_THRESHOLD:
                new_state = "PAUSED" if self.total_keystrokes > 0 else "INACTIVE"
            else:
                # Check recent IKI rhythm
                if self.recent_ikis:
                    recent_avg_iki = sum(list(self.recent_ikis)[-5:]) / min(5, len(self.recent_ikis))
                else:
                    recent_avg_iki = 0.3

                if recent_avg_iki < INTERMITTENT_IKI_THRESHOLD:
                    new_state = "ACTIVE TYPING"
                else:
                    new_state = "INTERMITTENT"

        if new_state != self.current_state:
            self.current_state = new_state
            self.state_start_time = now

    def get_metrics(self) -> TypingDynamicsData:
        """
        Polls current real-time metrics, advances timelines, and formats for display.
        Should be called every frame or tick (e.g., 30-50ms).
        """
        now = time.time()
        with self._lock:
            session_duration = max(0.1, now - self.session_start)

            # 1. Update activity state if user has paused without typing
            if self.last_press_time is not None:
                idle_since_last_key = now - self.last_press_time
                if idle_since_last_key >= BURST_PAUSE_THRESHOLD:
                    if self.is_in_burst:
                        self._finish_current_burst()
                if idle_since_last_key >= INACTIVITY_THRESHOLD:
                    if self.current_state != "PAUSED":
                        self.current_state = "PAUSED"
                        self.state_start_time = now
                    if not self.is_paused_idle:
                        self.is_paused_idle = True
                        self.total_pauses_count += 1
                        self.rolling_pauses.append(now)
                        if self.current_interval_marker == "NONE":
                            self.current_interval_marker = "PAUSE"
                current_pause_sec = max(0.0, idle_since_last_key)
            else:
                current_pause_sec = max(0.0, session_duration)

            # 2. Rolling Window Cleanups (60-second window)
            cutoff_60s = now - ROLLING_WINDOW_SEC
            while self.rolling_keystrokes and self.rolling_keystrokes[0] < cutoff_60s:
                self.rolling_keystrokes.popleft()

            while self.rolling_pauses and self.rolling_pauses[0] < cutoff_60s:
                self.rolling_pauses.popleft()

            # 3. Frequency & Speed Calculations
            kpm_rolling = len(self.rolling_keystrokes)
            gross_wpm = round(kpm_rolling / 5.0, 1)  # 5 keystrokes = 1 standard word

            session_minutes = session_duration / 60.0
            session_kpm = self.total_keystrokes / session_minutes
            session_wpm = round(session_kpm / 5.0, 1)

            # 4. Inter-Key Interval & Consistency
            if self.recent_ikis:
                mean_iki_sec = sum(self.recent_ikis) / len(self.recent_ikis)
                mean_iki_ms = int(mean_iki_sec * 1000)
                current_iki_ms = int(self.recent_ikis[-1] * 1000)

                # Standard deviation & Coefficient of Variation
                variance = sum((x - mean_iki_sec) ** 2 for x in self.recent_ikis) / len(self.recent_ikis)
                std_iki = math.sqrt(variance)
                cv = (std_iki / mean_iki_sec) if mean_iki_sec > 0 else 0.0

                if cv < 0.35:
                    consistency_label = "STEADY"
                elif cv < 0.65:
                    consistency_label = "MODERATE"
                else:
                    consistency_label = "VARIABLE"
            else:
                mean_iki_ms = 0
                current_iki_ms = 0
                cv = 0.0
                consistency_label = "STEADY"

            # 5. Pauses
            total_pauses = self.total_pauses_count
            avg_pause_sec = (sum(self.pause_durations) / len(self.pause_durations)) if self.pause_durations else 0.0
            pause_freq_str = f"{len(self.rolling_pauses)}/min"

            # 6. Error Correction & Persistence
            del_rate_pct = (self.total_deletions / max(1, self.total_keystrokes)) * 100.0
            avg_time_to_corr = (
                sum(self.time_to_first_corr_list) / len(self.time_to_first_corr_list)
            ) if self.time_to_first_corr_list else 0.0

            resumption_rate = (
                (self.post_correction_resumptions / max(1, self.deletion_bursts_count)) * 100.0
            ) if self.deletion_bursts_count > 0 else 100.0

            # 7. Timeline Graph Sample Update
            if now - self.last_timeline_sample_time >= TIMELINE_SAMPLE_INTERVAL:
                dt = max(0.1, now - self.last_timeline_sample_time)
                keys_per_sec = self.accumulated_keys_in_interval / dt
                instant_wpm = (keys_per_sec * 60.0) / 5.0
                norm_intensity = min(1.0, instant_wpm / 80.0)
                has_backspace = (self.accumulated_deletions_in_interval > 0)
                backspace_count = self.accumulated_deletions_in_interval

                self.timeline_data.append(
                    GraphPoint(
                        wpm=round(instant_wpm, 1),
                        intensity=round(norm_intensity, 3),
                        has_backspace=has_backspace,
                        backspace_count=backspace_count,
                    )
                )
                self.accumulated_keys_in_interval = 0
                self.accumulated_deletions_in_interval = 0
                self.current_interval_marker = "NONE"
                self.last_timeline_sample_time = now

            # 8. Time in State & Duration Formatting
            time_in_state = now - self.state_start_time

            return TypingDynamicsData(
                activity_state=self.current_state,
                time_in_state_str=self._format_duration(time_in_state),
                is_in_burst=self.is_in_burst,
                current_burst_keys=self.current_burst_keys,
                total_bursts_count=self.total_bursts_count,
                gross_wpm=gross_wpm,
                session_wpm=session_wpm,
                keystrokes_per_min=kpm_rolling,
                total_keystrokes=self.total_keystrokes,
                current_iki_ms=current_iki_ms,
                mean_iki_ms=mean_iki_ms,
                consistency_label=consistency_label,
                consistency_cv=round(cv, 2),
                current_pause_sec=round(current_pause_sec, 1),
                total_pauses_count=total_pauses,
                pause_frequency_str=pause_freq_str,
                avg_pause_sec=round(avg_pause_sec, 1),
                total_deletions=self.total_deletions,
                deletion_rate_pct=round(del_rate_pct, 1),
                deletion_bursts_count=self.deletion_bursts_count,
                max_deletion_burst=self.max_deletion_burst,
                avg_time_to_correction_sec=round(avg_time_to_corr, 2),
                revision_cycles_count=self.revision_cycles_count,
                post_correction_resumptions=self.post_correction_resumptions,
                resumption_rate_pct=round(resumption_rate, 1),
                session_duration_str=self._format_duration(session_duration),
                last_event_time_str=self.last_event_timestamp_str,
                timeline_history=list(self.timeline_data),
            )

    def reset(self) -> None:
        """Resets all metrics and timers to initial state (Thread-safe)."""
        with self._lock:
            self.__init__()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
