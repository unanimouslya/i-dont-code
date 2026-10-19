"""
Typing Dynamics Module - Standalone Extension HUD
Focus Detection Model for Students - Behavioral Sensing Module

Runs as a lightweight, floating Always-On-Top HUD overlay.
Detects keystrokes globally across Windows (Google Docs, Word, Chrome, Notion, etc.)
without storing typed text. Features a real-time speed curve with bright red dots
for backspaces/deletions.
"""

import sys
import time
from typing import List
import cv2
import numpy as np
from pynput import keyboard

from typing_dynamics import GraphPoint, TypingDynamicsAnalyzer, TypingDynamicsData


# ==============================================================================
# CONFIGURABLE PARAMETERS & STYLING (Matching Project 1 Aesthetic)
# ==============================================================================
WINDOW_TITLE = "Typing Dynamics HUD"

# Panel Dimensions
PANEL_W = 280
PANEL_H = 465

# Color Palette (BGR for OpenCV)
COLOR_BG = (22, 24, 30)            # Deep slate card background
COLOR_BORDER = (55, 60, 72)        # Card border outline
COLOR_DIVIDER = (45, 50, 60)       # Section divider line
COLOR_GRAPH_BG = (14, 16, 20)      # Graph inner background
COLOR_GRAPH_BORDER = (40, 45, 55)  # Graph inner border

# Text & Accent Colors (BGR)
COLOR_WHITE = (245, 245, 245)      # Primary text / values
COLOR_GRAY = (165, 170, 180)       # Secondary labels
COLOR_DIM = (105, 110, 120)        # Dim helper text
COLOR_GREEN = (75, 225, 125)       # Active, steady, persistence
COLOR_AMBER = (60, 180, 255)       # Intermittent, pauses, moderate
COLOR_CYAN = (245, 195, 60)        # Speed, bursts, trendline
COLOR_RED = (60, 60, 245)          # Deletions, backspace dots, variable
COLOR_PURPLE = (235, 140, 170)     # Revision cycles


class TypingHUDApp:
    def __init__(self):
        self.analyzer = TypingDynamicsAnalyzer()
        self.running = True
        self.listener = None
        self._start_global_listener()

    def _start_global_listener(self):
        """Starts asynchronous global keyboard listener across the operating system."""
        def on_press(key):
            try:
                if key == keyboard.Key.backspace:
                    cat = "BACKSPACE"
                elif key == keyboard.Key.delete:
                    cat = "DELETE"
                elif key == keyboard.Key.space:
                    cat = "SPACE"
                elif key in (keyboard.Key.enter,):
                    cat = "ENTER"
                elif key in (
                    keyboard.Key.shift, keyboard.Key.shift_r,
                    keyboard.Key.ctrl, keyboard.Key.ctrl_r,
                    keyboard.Key.alt, keyboard.Key.alt_r,
                    keyboard.Key.caps_lock
                ):
                    cat = "MOD"
                elif key in (
                    keyboard.Key.up, keyboard.Key.down,
                    keyboard.Key.left, keyboard.Key.right,
                    keyboard.Key.home, keyboard.Key.end,
                    keyboard.Key.page_up, keyboard.Key.page_down
                ):
                    cat = "NAV"
                else:
                    cat = "CHAR"
            except Exception:
                cat = "CHAR"

            self.analyzer.record_key_event("PRESS", cat)

        def on_release(key):
            try:
                if key in (keyboard.Key.backspace, keyboard.Key.delete):
                    cat = "BACKSPACE"
                else:
                    cat = "CHAR"
            except Exception:
                cat = "CHAR"
            self.analyzer.record_key_event("RELEASE", cat)

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()

    def render_hud(self, data: TypingDynamicsData) -> np.ndarray:
        """Draws the compact floating HUD card matching the posture-movement module aesthetic."""
        frame = np.full((PANEL_H, PANEL_W, 3), COLOR_BG, dtype=np.uint8)

        # Outer border
        cv2.rectangle(frame, (0, 0), (PANEL_W - 1, PANEL_H - 1), COLOR_BORDER, 1)

        x1 = 12
        x2 = PANEL_W - 12

        # ----------------------------------------------------------------------
        # Header
        # ----------------------------------------------------------------------
        y = 24
        cv2.putText(frame, "TYPING DYNAMICS", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 2, cv2.LINE_AA)
        
        # Live activity dot in header
        if data.activity_state == "ACTIVE TYPING":
            dot_col = COLOR_GREEN
        elif data.activity_state == "INTERMITTENT":
            dot_col = COLOR_AMBER
        elif data.activity_state == "PAUSED":
            dot_col = COLOR_GRAY
        else:
            dot_col = COLOR_DIM
        cv2.circle(frame, (x2 - 8, y - 5), 4, dot_col, -1, cv2.LINE_AA)

        y += 8
        cv2.line(frame, (x1, y), (x2, y), COLOR_DIVIDER, 1)

        # ----------------------------------------------------------------------
        # Section 1: Activity & Speed
        # ----------------------------------------------------------------------
        y += 20
        cv2.putText(frame, "ACTIVITY:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        act_col = (
            COLOR_GREEN if data.activity_state == "ACTIVE TYPING"
            else (COLOR_AMBER if data.activity_state == "INTERMITTENT" else COLOR_GRAY)
        )
        cv2.putText(frame, data.activity_state, (x1 + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, act_col, 2, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "CADENCE:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        cad_col = (
            COLOR_GREEN if data.consistency_label == "STEADY"
            else (COLOR_AMBER if data.consistency_label == "MODERATE" else COLOR_RED)
        )
        cv2.putText(frame, f"{data.consistency_label}", (x1 + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, cad_col, 2, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "SPEED:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{data.gross_wpm:.1f} WPM", (x1 + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN, 2, cv2.LINE_AA)

        y += 9
        cv2.line(frame, (x1, y), (x2, y), COLOR_DIVIDER, 1)

        # ----------------------------------------------------------------------
        # Section 2: Current State & Dwell Time
        # ----------------------------------------------------------------------
        y += 18
        cv2.putText(frame, "CURRENT STATE:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        state_col = COLOR_GREEN if data.activity_state in ("ACTIVE TYPING", "INTERMITTENT") else COLOR_DIM
        state_label = "ACTIVE" if data.activity_state in ("ACTIVE TYPING", "INTERMITTENT") else "PAUSED"
        cv2.putText(frame, state_label, (x1 + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, state_col, 2, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "TIME IN STATE:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, data.time_in_state_str, (x1 + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_WHITE, 2, cv2.LINE_AA)

        y += 9
        cv2.line(frame, (x1, y), (x2, y), COLOR_DIVIDER, 1)

        # ----------------------------------------------------------------------
        # Section 3: Dynamics & Rhythm
        # ----------------------------------------------------------------------
        y += 18
        cv2.putText(frame, "BURST:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        if data.is_in_burst:
            cv2.putText(frame, f"{data.current_burst_keys} keys", (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_CYAN, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "SETTLED", (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_DIM, 1, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "FREQUENCY:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{data.keystrokes_per_min}/min", (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_WHITE, 2, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "AVG IKI:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{data.mean_iki_ms} ms", (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_WHITE, 1, cv2.LINE_AA)

        y += 9
        cv2.line(frame, (x1, y), (x2, y), COLOR_DIVIDER, 1)

        # ----------------------------------------------------------------------
        # Section 4: Errors & Persistence
        # ----------------------------------------------------------------------
        y += 18
        cv2.putText(frame, "PAUSE:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        pause_str = f"{data.current_pause_sec:.1f}s ({data.total_pauses_count} total)"
        pause_col = COLOR_AMBER if data.current_pause_sec > 1.4 else COLOR_WHITE
        cv2.putText(frame, pause_str, (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, pause_col, 1, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "DELETIONS:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        del_str = f"{data.total_deletions} ({data.deletion_rate_pct:.1f}%)"
        del_col = COLOR_RED if data.total_deletions > 0 else COLOR_WHITE
        cv2.putText(frame, del_str, (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, del_col, 2, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "REVISIONS:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{data.revision_cycles_count} cycles", (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_PURPLE, 1, cv2.LINE_AA)

        y += 19
        cv2.putText(frame, "PERSISTENCE:", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_GRAY, 1, cv2.LINE_AA)
        pers_str = f"{data.resumption_rate_pct:.0f}% ({data.post_correction_resumptions}/{max(1, data.deletion_bursts_count)})"
        cv2.putText(frame, pers_str, (x1 + 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_GREEN, 2, cv2.LINE_AA)

        y += 9
        cv2.line(frame, (x1, y), (x2, y), COLOR_DIVIDER, 1)

        # ----------------------------------------------------------------------
        # Section 5: Real-Time Speed & Backspace Error Graph (45s)
        # ----------------------------------------------------------------------
        y += 16
        cv2.putText(frame, "SPEED & ERROR GRAPH (45s):", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_GRAY, 1, cv2.LINE_AA)

        graph_x = x1
        graph_y = y + 8
        graph_w = x2 - x1
        graph_h = 58

        self._draw_graph(frame, data.timeline_history, graph_x, graph_y, graph_w, graph_h, data.gross_wpm)

        # Graph Legend
        legend_y = graph_y + graph_h + 15
        # Blue line for speed
        cv2.line(frame, (graph_x + 2, legend_y - 3), (graph_x + 18, legend_y - 3), COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "SPEED", (graph_x + 22, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_GRAY, 1, cv2.LINE_AA)

        # Red dot for backspace
        dot_x = graph_x + 85
        cv2.circle(frame, (dot_x, legend_y - 3), 4, (20, 20, 30), -1, cv2.LINE_AA)
        cv2.circle(frame, (dot_x, legend_y - 3), 3, COLOR_RED, -1, cv2.LINE_AA)
        cv2.putText(frame, "BACKSPACE", (dot_x + 8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_GRAY, 1, cv2.LINE_AA)

        # Hotkeys help
        cv2.putText(frame, "[r] Reset | [q] Exit", (x2 - 105, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_DIM, 1, cv2.LINE_AA)

        return frame

    def _draw_graph(self, frame: np.ndarray, history: List[GraphPoint], gx: int, gy: int, gw: int, gh: int, current_wpm: float):
        """
        Renders a clean, continuous line graph of typing speed with prominent
        bright red dots plotted at every sample interval where backspace occurred.
        """
        # Graph background card
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), COLOR_GRAPH_BG, -1)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), COLOR_GRAPH_BORDER, 1)

        # Subtle gridlines (50% level and baseline)
        mid_y = gy + (gh // 2)
        cv2.line(frame, (gx + 1, mid_y), (gx + gw - 1, mid_y), (25, 28, 35), 1, cv2.LINE_AA)
        base_y = gy + gh - 4

        if not history:
            return

        n = len(history)
        step_x = (gw - 8) / max(1, n - 1)

        pts = []
        backspace_pts = []

        for i, pt in enumerate(history):
            px = int(gx + 4 + (i * step_x))
            # Height scales with intensity (up to 80 WPM)
            py = int(base_y - (pt.intensity * (gh - 10)))
            py = max(gy + 4, min(base_y, py))
            pts.append((px, py))

            if pt.has_backspace:
                backspace_pts.append((px, py, pt.backspace_count))

        # 1. Subtle shaded area under curve
        if len(pts) >= 2:
            poly_pts = [(pts[0][0], base_y)] + pts + [(pts[-1][0], base_y)]
            poly_arr = np.array(poly_pts, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_arr], (70, 55, 18))  # Dark subtle cyan tint
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # 2. Continuous speed line curve
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], COLOR_CYAN, 2, cv2.LINE_AA)

        # 3. Bright Red Dots for Backspaces
        for bx, by, count in backspace_pts:
            # Contrast border outline
            radius = 5 if count >= 3 else 4
            cv2.circle(frame, (bx, by), radius + 1, (20, 20, 30), -1, cv2.LINE_AA)
            # Bright Red core dot
            cv2.circle(frame, (bx, by), radius, COLOR_RED, -1, cv2.LINE_AA)

        # Current live WPM tag in top-right of graph
        speed_tag = f"{current_wpm:.0f} WPM"
        cv2.putText(frame, speed_tag, (gx + gw - 54, gy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_CYAN, 1, cv2.LINE_AA)

    def run(self):
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        # Set Always-On-Top so user can type in Word, Docs, or Chrome while monitoring
        try:
            cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        print("Typing Dynamics Extension HUD running.")
        print("Type in any application (Word, Google Docs, Chrome, etc.) to observe live dynamics.")
        print("Press 'r' to reset stats | Press 'q' or ESC on the HUD window to quit.")

        while self.running:
            data = self.analyzer.get_metrics()
            frame = self.render_hud(data)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(35) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                break
            elif key in (ord('r'), ord('R')):    # 'r' to reset
                self.analyzer.reset()

            # Check if window was closed via [X] button
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        if self.listener:
            self.listener.stop()
        cv2.destroyAllWindows()


def main():
    app = TypingHUDApp()
    app.run()


if __name__ == "__main__":
    main()
