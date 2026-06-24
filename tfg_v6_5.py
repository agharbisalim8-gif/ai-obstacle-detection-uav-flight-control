import csv
import json
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from djitellopy import Tello
from ultralytics import YOLO

# Profile
PROFILE_NAME = "V6_5_HEAD_ZONE_UPWARD_AVOIDANCE"
PROFILE_NOTE = (
    "Batch-labelled experimental version with upward escape for a centred person/head. "
    "Downward branch disabled. Operator labels: Y, F, U."
)


# Configuration
MODEL_PATH = "yolo11n.pt"
BASE_OUTPUT_DIR = "tello_quantitative_trials_v6_5_bidirectional_vertical_escape"
WINDOW_NAME = "TFG V6.5 - Bidirectional Vertical Escape Test"

TARGET_CLASSES = {"person", "car", "truck", "bus", "bicycle", "motorcycle"}
CONF_THRES = 0.35
SHOW_NON_TARGETS = False


# Safety corridor
CENTER_X_MIN = 0.32
CENTER_X_MAX = 0.68
SAFETY_X_MIN = 0.28
SAFETY_X_MAX = 0.72
EXIT_MARGIN = 0.00


# Decision thresholds
AREA_WARNING = 0.550
AREA_TRIGGER = 0.700
AREA_HARD_STOP = 0.920


GROWTH_TRIGGER = 0.350
AREA_HISTORY_LEN = 6
CONFIRM_FRAMES_REQUIRED = 3
EXIT_CORRIDOR_FRAMES_REQUIRED = 2
LOST_CLEAR_FRAMES_REQUIRED = 6
CLEAR_AREA_MAX = 0.025


BYPASS_RECHECK_AREA_MIN = 0.120


USE_TTC_TRIGGER = False
TTC_TRIGGER_SEC = 1.40
MIN_TTC_AREA = 0.180
MIN_AREA_GROWTH_RATE = 0.080


# Distance proxy
USE_DISTANCE_TRIGGER = True
REF_DISTANCE_M = 0.50
AREA_REF_AT_DIST = 0.800
TRIGGER_DISTANCE_M = 0.45
HARD_STOP_DISTANCE_M = 0.30


# Timing limits
TAKEOFF_STABILIZE_SEC = 2.0
MIN_BATTERY_PERCENT = 20
MIN_APPROACH_SEC = 1.20
MAX_TRIGGER_WAIT_SEC = 24.0
BRAKE_BEFORE_AVOID_SEC = 0.40
MIN_LATERAL_SEC = 0.35
MAX_LATERAL_SEC = 6.00


BYPASS_FORWARD_SEC = 8.00


RECENTER_SEC = 1.50


FINAL_FORWARD_SEC = 3.00


MIN_BYPASS_BEFORE_RECHECK_SEC = 1.20


# Bypass and recenter
USE_DISTANCE_BASED_BYPASS = True
BYPASS_TARGET_DISTANCE_M = 2.50
FB_MPS_AT_RC20 = 0.30
MIN_BYPASS_FORWARD_SEC = 3.00
MAX_BYPASS_FORWARD_SEC = 12.00


USE_EFFORT_BASED_RECENTER = True
RECENTER_COMPENSATION = 1.10
MIN_RECENTER_SEC = 1.50
MAX_RECENTER_SEC = 5.00


# Bypass mode
BYPASS_MODE = "straight"
BYPASS_DIAGONAL_LR = 0
BYPASS_DIAGONAL_SEC = 0.0


# Vertical avoidance
ENABLE_UPWARD_AVOIDANCE = True
UP_ONLY_FOR_CLASSES = {"person"}


UP_AREA_TRIGGER = 0.400
UP_DISTANCE_TRIGGER_M = 0.75
UP_MAX_AREA = 0.900


UP_CENTER_MAX_OFFSET = 0.22
UP_MIN_TOP_CLEARANCE = 0.18
UP_MAX_BBOX_HEIGHT_NORM = 0.88
UP_CLEAR_TOP_Y_NORM = 0.50
UP_LATERAL_EDGE_BIAS = 0.00


UP_HEAD_Y1_MAX = 0.25
UP_HEAD_CENTER_Y_MAX = 0.55


UP_MAX_HEIGHT_CM = 170


RC_UP_AVOID = 22
UP_AVOID_MIN_SEC = 0.70
UP_AVOID_TARGET_SEC = 1.15
UP_AVOID_MAX_SEC = 1.80


USE_DOWN_RECENTER_AFTER_UP = False
DOWN_RECENTER_SEC = 0.60
RC_DOWN_RECENTER = 12


# Downward test branch
ENABLE_DOWNWARD_AVOIDANCE = False
DOWN_ONLY_FOR_CLASSES = {"person"}


DOWN_AREA_TRIGGER = 0.520
DOWN_DISTANCE_TRIGGER_M = 0.60
DOWN_MAX_AREA = 0.850
DOWN_CENTER_MAX_OFFSET = 0.16
DOWN_MAX_CENTER_Y = 0.48
DOWN_MIN_BOTTOM_CLEARANCE = 0.18
DOWN_MAX_BBOX_HEIGHT_NORM = 0.82
DOWN_CLEAR_BOTTOM_Y_NORM = 0.50
DOWN_LATERAL_EDGE_BIAS = 0.09


DOWN_MIN_HEIGHT_CM = 90


RC_DOWN_AVOID = 18
DOWN_AVOID_MIN_SEC = 0.50
DOWN_AVOID_TARGET_SEC = 0.85
DOWN_AVOID_MAX_SEC = 1.30


USE_UP_RECENTER_AFTER_DOWN = False
UP_RECENTER_AFTER_DOWN_SEC = 0.60
RC_UP_RECENTER_AFTER_DOWN = 12


# RC command values
RC_MANUAL_SPEED = 28
RC_YAW_SPEED = 35
RC_UPDOWN_SPEED = 28
RC_APPROACH_FORWARD = 14
RC_BYPASS_FORWARD = 22
RC_FINAL_FORWARD = 18
RC_LATERAL_BASE = 24
RC_LATERAL_STRONG = 30
RC_LATERAL_MAX = 36
RC_RECENTER = 16
CONTROL_ALPHA = 0.65
EMERGENCY_BACK_SPEED = -8

# Output logging
SAVE_ANNOTATED_VIDEO = True
SAVE_RAW_VIDEO = True
SAVE_PERIODIC_FRAMES = True
PERIODIC_FRAME_INTERVAL_SEC = 0.7
VIDEO_FPS = 20
TELEMETRY_LOG_INTERVAL_SEC = 0.3

# Keyboard codes
ARROW_LEFT = 2424832
ARROW_UP = 2490368
ARROW_RIGHT = 2555904
ARROW_DOWN = 2621440


# Utility functions
def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def make_run_dir(base_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts}"
    (run_dir / "frames_raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "frames_annotated").mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_call(fn, default=-1):
    try:
        return fn()
    except Exception:
        return default


def open_csv_writer(path: Path, header):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(header)
    f.flush()
    return f, writer


def safe_stop_rc(tello: Tello):
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass


def smooth(prev, new, alpha=CONTROL_ALPHA):
    return int(alpha * prev + (1.0 - alpha) * new)


def bbox_intersects_safety_corridor(det) -> bool:
    if det is None:
        return False
    return (det["x2_norm"] >= SAFETY_X_MIN) and (det["x1_norm"] <= SAFETY_X_MAX)


def bbox_outside_safety_corridor(det) -> bool:


    if det is None:
        return False
    left_clear = det["x2_norm"] < (SAFETY_X_MIN - EXIT_MARGIN)
    right_clear = det["x1_norm"] > (SAFETY_X_MAX + EXIT_MARGIN)
    return left_clear or right_clear


def estimate_distance_from_area(area_norm: float):
    if area_norm <= 1e-6 or AREA_REF_AT_DIST <= 1e-6:
        return None
    return REF_DISTANCE_M * (AREA_REF_AT_DIST / area_norm) ** 0.5


def estimate_forward_mps(rc_forward: int) -> float:


    return FB_MPS_AT_RC20 * (abs(rc_forward) / 20.0)


def choose_avoid_direction(primary) -> int:


    if primary is None:
        return +1
    return +1 if primary["center_x_norm"] < 0.50 else -1


def compute_area_growth_rate(area_history):

    if len(area_history) < 2:
        return 0.0
    t0, a0 = area_history[0]
    t1, a1 = area_history[-1]
    dt = max(1e-3, t1 - t0)
    return (a1 - a0) / dt


def compute_ttc_proxy(area_norm, area_growth_rate):

    if area_growth_rate <= 1e-6 or area_norm <= 1e-6:
        return None
    return area_norm / area_growth_rate


def bbox_height_norm(det) -> float:
    if det is None:
        return 0.0
    return max(0.0, det.get("y2_norm", 0.0) - det.get("y1_norm", 0.0))


def obstacle_below_vertical_clearance(det) -> bool:


    if det is None:
        return False
    return det.get("y1_norm", 0.0) >= UP_CLEAR_TOP_Y_NORM


def obstacle_above_vertical_clearance(det) -> bool:


    if det is None:
        return False
    return det.get("y2_norm", 1.0) <= DOWN_CLEAR_BOTTOM_Y_NORM


def choose_avoidance_strategy(primary, current_area, distance_est, height_cm):


    lateral_sign = choose_avoid_direction(primary)
    lateral_text = "right" if lateral_sign == +1 else "left"

    if primary is None:
        return "lateral", lateral_sign, lateral_text, "no_primary_detection"

    if not bbox_intersects_safety_corridor(primary):
        return "lateral", lateral_sign, lateral_text, "outside_safety_corridor"

    cx = primary["center_x_norm"]
    cy = primary.get("center_y_norm", 0.5)
    y1n = primary.get("y1_norm", 0.0)
    y2n = primary.get("y2_norm", 1.0)
    h_norm = bbox_height_norm(primary)

    lateral_easy = (cx < CENTER_X_MIN + UP_LATERAL_EDGE_BIAS) or (cx > CENTER_X_MAX - UP_LATERAL_EDGE_BIAS)


    if ENABLE_DOWNWARD_AVOIDANCE and primary["class_name"] in DOWN_ONLY_FOR_CLASSES:
        down_close_enough = (
            current_area >= DOWN_AREA_TRIGGER or
            (distance_est is not None and distance_est <= DOWN_DISTANCE_TRIGGER_M)
        )
        down_not_too_close = current_area <= DOWN_MAX_AREA
        down_centred = abs(cx - 0.50) <= DOWN_CENTER_MAX_OFFSET
        enough_space_below = (1.0 - y2n) >= DOWN_MIN_BOTTOM_CLEARANCE
        obstacle_high = cy <= DOWN_MAX_CENTER_Y
        down_not_full_height = h_norm <= DOWN_MAX_BBOX_HEIGHT_NORM

        if height_cm is None or height_cm < 0:
            down_altitude_ok = False
            down_altitude_note = "height_unknown_down_disabled"
        else:
            down_altitude_ok = height_cm >= DOWN_MIN_HEIGHT_CM
            down_altitude_note = f"height={height_cm}cm"

        if (down_close_enough and down_not_too_close and down_centred and
                enough_space_below and obstacle_high and down_not_full_height and
                down_altitude_ok and not lateral_easy):
            reason = (
                f"down_selected A={current_area:.3f}, cx={cx:.2f}, cy={cy:.2f}, "
                f"y2={y2n:.2f}, h={h_norm:.2f}, {down_altitude_note}"
            )
            return "down", 0, "down", reason


    if ENABLE_UPWARD_AVOIDANCE and primary["class_name"] in UP_ONLY_FOR_CLASSES:
        up_close_enough = (
            current_area >= UP_AREA_TRIGGER or
            (distance_est is not None and distance_est <= UP_DISTANCE_TRIGGER_M)
        )
        up_not_too_close = current_area <= UP_MAX_AREA


        up_centred = abs(cx - 0.50) <= UP_CENTER_MAX_OFFSET


        head_or_upper_zone = (
            y1n <= UP_HEAD_Y1_MAX or
            cy <= UP_HEAD_CENTER_Y_MAX
        )

        up_not_full_height = h_norm <= UP_MAX_BBOX_HEIGHT_NORM

        if height_cm is None or height_cm < 0:
            up_altitude_ok = True
            up_altitude_note = "height_unknown"
        else:
            up_altitude_ok = height_cm <= UP_MAX_HEIGHT_CM
            up_altitude_note = f"height={height_cm}cm"

        if (
            up_close_enough and
            up_not_too_close and
            up_centred and
            head_or_upper_zone and
            up_not_full_height and
            up_altitude_ok
        ):
            reason = (
                f"up_selected_HEAD_ZONE A={current_area:.3f}, cx={cx:.2f}, cy={cy:.2f}, "
                f"y1={y1n:.2f}, y2={y2n:.2f}, h={h_norm:.2f}, {up_altitude_note}"
            )
            return "up", 0, "up", reason

    reason = (
        f"lateral_selected A={current_area:.3f}, cx={cx:.2f}, cy={cy:.2f}, "
        f"y1={y1n:.2f}, y2={y2n:.2f}, h={h_norm:.2f}, lateral_easy={lateral_easy}"
    )
    return "lateral", lateral_sign, lateral_text, reason

def draw_overlay(
    frame,
    flying,
    state,
    scenario_name,
    attempt_active,
    primary,
    distance_est,
    ttc_proxy,
    danger,
    warning,
    exit_counter,
    lost_counter,
    lateral_effort,
    recenter_target_sec,
    bypass_distance_est_m,
    avoidance_mode="none",
    vertical_effort=0.0,
    up_duration=0.0,
):
    lines = [
        "T Takeoff | L Land | Q Quit",
        "Arrows move | W/S up-down | A/D yaw",
        "1/2/3/4/5 scenario | Enter start attempt | X abort",
        f"Profile: {PROFILE_NAME}",
        f"State: {state}",
        f"Scenario: {scenario_name}",
        f"Attempt: {'ON' if attempt_active else 'OFF'}",
        f"Warning: {warning} | Danger: {danger} | Exit frames: {exit_counter} | Lost frames: {lost_counter}",
        f"Lat effort: {lateral_effort:.1f} | Recenter target: {recenter_target_sec:.2f}s",
        f"Bypass est: {bypass_distance_est_m:.2f}m / target {BYPASS_TARGET_DISTANCE_M:.2f}m",
        f"Avoid mode: {avoidance_mode} | Vertical effort: {vertical_effort:.1f} | Vertical sec: {up_duration:.2f}",
        "After attempt: Y success | F failure | U unclear",
    ]
    if primary is not None:
        dist_txt = f" d~{distance_est:.2f}m" if distance_est is not None else " d~NA"
        ttc_txt = f" ttc~{ttc_proxy:.2f}s" if ttc_proxy is not None else " ttc~NA"
        lines.append(
            f"Primary: {primary['class_name']} A={primary['bbox_area_norm']:.3f} "
            f"cx={primary['center_x_norm']:.2f} cy={primary['center_y_norm']:.2f} "
            f"x=[{primary['x1_norm']:.2f},{primary['x2_norm']:.2f}] "
            f"y=[{primary.get('y1_norm', 0.0):.2f},{primary.get('y2_norm', 0.0):.2f}]"
            + dist_txt + ttc_txt
        )
    if not flying:
        lines.append("NEXT: Press T to take off")

    y = 24
    for txt in lines:
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        y += 21


# Main program
def main():
    model = YOLO(MODEL_PATH)
    run_dir = make_run_dir(BASE_OUTPUT_DIR)

    config = {
        "profile_name": PROFILE_NAME,
        "profile_note": PROFILE_NOTE,
        "model_path": MODEL_PATH,
        "conf_thres": CONF_THRES,
        "control_style": "CLEARANCE_AROUND_OBSTACLE_RC_FSM_PROFILED",
        "architecture": "Perception-Decision-Actuation",
        "center_x_min": CENTER_X_MIN,
        "center_x_max": CENTER_X_MAX,
        "safety_x_min": SAFETY_X_MIN,
        "safety_x_max": SAFETY_X_MAX,
        "exit_margin": EXIT_MARGIN,
        "area_warning": AREA_WARNING,
        "area_trigger": AREA_TRIGGER,
        "area_hard_stop": AREA_HARD_STOP,
        "growth_trigger": GROWTH_TRIGGER,
        "confirm_frames_required": CONFIRM_FRAMES_REQUIRED,
        "exit_corridor_frames_required": EXIT_CORRIDOR_FRAMES_REQUIRED,
        "lost_clear_frames_required": LOST_CLEAR_FRAMES_REQUIRED,
        "clear_area_max": CLEAR_AREA_MAX,
        "use_ttc_trigger": USE_TTC_TRIGGER,
        "ttc_trigger_sec": TTC_TRIGGER_SEC,
        "min_ttc_area": MIN_TTC_AREA,
        "min_area_growth_rate": MIN_AREA_GROWTH_RATE,
        "use_distance_trigger": USE_DISTANCE_TRIGGER,
        "ref_distance_m": REF_DISTANCE_M,
        "area_ref_at_dist": AREA_REF_AT_DIST,
        "trigger_distance_m": TRIGGER_DISTANCE_M,
        "hard_stop_distance_m": HARD_STOP_DISTANCE_M,
        "min_approach_sec": MIN_APPROACH_SEC,
        "bypass_forward_sec_fallback": BYPASS_FORWARD_SEC,
        "use_distance_based_bypass": USE_DISTANCE_BASED_BYPASS,
        "bypass_target_distance_m": BYPASS_TARGET_DISTANCE_M,
        "fb_mps_at_rc20": FB_MPS_AT_RC20,
        "use_effort_based_recenter": USE_EFFORT_BASED_RECENTER,
        "recenter_compensation": RECENTER_COMPENSATION,
        "min_recenter_sec": MIN_RECENTER_SEC,
        "max_recenter_sec": MAX_RECENTER_SEC,
        "bypass_recheck_area_min": BYPASS_RECHECK_AREA_MIN,
        "bypass_mode": BYPASS_MODE,
        "enable_upward_avoidance": ENABLE_UPWARD_AVOIDANCE,
        "up_area_trigger": UP_AREA_TRIGGER,
        "up_distance_trigger_m": UP_DISTANCE_TRIGGER_M,
        "up_min_top_clearance": UP_MIN_TOP_CLEARANCE,
        "up_clear_top_y_norm": UP_CLEAR_TOP_Y_NORM,
        "up_center_max_offset": UP_CENTER_MAX_OFFSET,
        "up_head_y1_max": UP_HEAD_Y1_MAX,
        "up_head_center_y_max": UP_HEAD_CENTER_Y_MAX,
        "up_lateral_edge_bias": UP_LATERAL_EDGE_BIAS,
        "up_max_height_cm": UP_MAX_HEIGHT_CM,
        "rc_up_avoid": RC_UP_AVOID,
        "up_avoid_target_sec": UP_AVOID_TARGET_SEC,
        "use_down_recenter_after_up": USE_DOWN_RECENTER_AFTER_UP,
        "enable_downward_avoidance": ENABLE_DOWNWARD_AVOIDANCE,
        "down_area_trigger": DOWN_AREA_TRIGGER,
        "down_distance_trigger_m": DOWN_DISTANCE_TRIGGER_M,
        "down_min_bottom_clearance": DOWN_MIN_BOTTOM_CLEARANCE,
        "down_clear_bottom_y_norm": DOWN_CLEAR_BOTTOM_Y_NORM,
        "down_center_max_offset": DOWN_CENTER_MAX_OFFSET,
        "down_max_center_y": DOWN_MAX_CENTER_Y,
        "down_min_height_cm": DOWN_MIN_HEIGHT_CM,
        "rc_down_avoid": RC_DOWN_AVOID,
        "down_avoid_target_sec": DOWN_AVOID_TARGET_SEC,
        "use_up_recenter_after_down": USE_UP_RECENTER_AFTER_DOWN,
        "created_at": now_iso(),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    events_f, events_writer = open_csv_writer(
        run_dir / "events.csv",
        ["timestamp", "frame_idx", "event", "details"]
    )
    attempts_f, attempts_writer = open_csv_writer(
        run_dir / "attempts.csv",
        [
            "attempt_id", "scenario", "profile", "timestamp_start", "timestamp_end",
            "battery_start", "battery_end", "height_start_cm", "height_end_cm",
            "obstacle_class_expected", "detection_success", "triggered", "avoidance_success",
            "false_trigger", "first_detection_frame", "trigger_frame", "clearance_frame",
            "trigger_area_norm", "trigger_growth", "trigger_growth_rate", "trigger_ttc_proxy_sec",
            "trigger_cx", "trigger_distance_est_m", "avoid_direction", "avoidance_mode",
            "vertical_duration_sec", "vertical_effort", "lateral_duration_sec",
            "lateral_effort", "bypass_duration_sec", "bypass_distance_est_m",
            "recenter_target_sec", "notes"
        ]
    )
    detections_f, detections_writer = open_csv_writer(
        run_dir / "detections.csv",
        [
            "timestamp", "frame_idx", "attempt_id", "scenario", "profile", "state",
            "det_idx", "class_name", "confidence", "x1", "y1", "x2", "y2",
            "x1_norm", "x2_norm", "bbox_area_norm", "center_x_norm", "center_y_norm",
            "distance_est_m", "is_target", "is_primary", "intersects_safety_corridor",
            "y1_norm", "y2_norm", "bbox_height_norm", "below_up_clearance", "above_down_clearance"
        ]
    )
    telemetry_f, telemetry_writer = open_csv_writer(
        run_dir / "telemetry.csv",
        [
            "timestamp", "frame_idx", "attempt_id", "scenario", "profile",
            "battery", "height_cm", "tof_cm", "yaw", "pitch", "roll",
            "vgx", "vgy", "vgz", "state", "flying", "attempt_active",
            "lateral_effort", "bypass_distance_est_m", "recenter_target_sec",
            "avoidance_mode", "vertical_effort", "vertical_duration_sec"
        ]
    )
    labels_f, labels_writer = open_csv_writer(
        run_dir / "operator_trial_labels.csv",
        [
            "timestamp", "attempt_id", "scenario", "profile",
            "operator_label", "operator_success", "auto_avoidance_success",
            "triggered", "detection_success", "notes"
        ]
    )

    def log_event(frame_idx, event, details=""):
        print(f"[EVENT] {event} | {details}")
        events_writer.writerow([now_iso(), frame_idx, event, details])
        events_f.flush()

    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    print(f"[INFO] Profile: {PROFILE_NAME}")
    print(f"[INFO] Battery: {battery}%")
    if battery < MIN_BATTERY_PERCENT:
        raise RuntimeError(f"Battery too low for safe testing: {battery}%")

    tello.streamon()
    frame_read = tello.get_frame_read()
    time.sleep(1.5)

    raw_writer = None
    ann_writer = None
    frame_idx = 0
    last_frame_save = 0.0
    last_telemetry_log = 0.0

    flying = False
    state = "IDLE"
    attempt_id = 0
    attempt_active = False
    scenario_name = "centered_person"
    expected_obstacle_class = "person"
    attempt_start_ts = ""
    attempt_start_time = 0.0
    attempt_notes = ""
    battery_start = -1
    height_start = -1
    first_detection_frame = -1
    trigger_frame = -1
    clearance_frame = -1
    trigger_area = 0.0
    trigger_growth = 0.0
    trigger_growth_rate = 0.0
    trigger_ttc_proxy = -1.0
    trigger_cx = 0.0
    trigger_distance_est = -1.0
    detection_success = False
    triggered = False
    avoidance_success = False
    false_trigger = False
    avoid_direction_text = "none"
    avoid_direction_sign = 0
    exit_counter = 0
    lost_counter = 0
    recenter_clear_counter = 0
    state_start_time = 0.0
    lateral_start_time = 0.0
    lateral_duration = 0.0
    bypass_start_time = 0.0
    bypass_duration = 0.0
    bypass_distance_est_m = 0.0
    lateral_effort = 0.0
    avoidance_mode = "none"
    up_start_time = 0.0
    up_duration = 0.0
    vertical_effort = 0.0
    recenter_target_sec = RECENTER_SEC
    last_motion_update_time = time.time()
    smoothed_fb = 0
    smoothed_lr = 0


    awaiting_operator_label = False
    pending_label_attempt_id = -1
    pending_label_auto_success = False
    pending_label_triggered = False
    pending_label_detection_success = False
    pending_label_notes = ""
    labelled_trials = 0
    labelled_successes = 0

    area_history = deque(maxlen=AREA_HISTORY_LEN)
    confirm_counter = 0

    def reset_smoothing():
        nonlocal smoothed_fb, smoothed_lr
        smoothed_fb = 0
        smoothed_lr = 0

    def stop_and_reset():
        safe_stop_rc(tello)
        reset_smoothing()

    def send_smooth_rc(forward_backward_target, left_right_target, up_down_target=0, yaw_target=0):
        nonlocal smoothed_fb, smoothed_lr
        smoothed_fb = smooth(smoothed_fb, forward_backward_target)
        smoothed_lr = smooth(smoothed_lr, left_right_target)
        tello.send_rc_control(int(smoothed_lr), int(smoothed_fb), int(up_down_target), int(yaw_target))

    def set_state(new_state, reason=""):
        nonlocal state, state_start_time, last_motion_update_time
        if state != new_state:
            state = new_state
            state_start_time = time.time()
            last_motion_update_time = time.time()
            log_event(frame_idx, f"state_{new_state}", reason)

    def reset_attempt_metrics():
        nonlocal attempt_notes, first_detection_frame, trigger_frame, clearance_frame
        nonlocal trigger_area, trigger_growth, trigger_growth_rate, trigger_ttc_proxy
        nonlocal trigger_cx, trigger_distance_est
        nonlocal detection_success, triggered, avoidance_success, false_trigger
        nonlocal avoid_direction_text, avoid_direction_sign, exit_counter, lost_counter
        nonlocal recenter_clear_counter, state_start_time, lateral_start_time
        nonlocal lateral_duration, bypass_start_time, bypass_duration, confirm_counter
        nonlocal bypass_distance_est_m, lateral_effort, avoidance_mode, up_start_time, up_duration
        nonlocal vertical_effort, recenter_target_sec, last_motion_update_time
        attempt_notes = ""
        first_detection_frame = -1
        trigger_frame = -1
        clearance_frame = -1
        trigger_area = 0.0
        trigger_growth = 0.0
        trigger_growth_rate = 0.0
        trigger_ttc_proxy = -1.0
        trigger_cx = 0.0
        trigger_distance_est = -1.0
        detection_success = False
        triggered = False
        avoidance_success = False
        false_trigger = False
        avoid_direction_text = "none"
        avoid_direction_sign = 0
        exit_counter = 0
        lost_counter = 0
        recenter_clear_counter = 0
        area_history.clear()
        confirm_counter = 0
        state_start_time = time.time()
        lateral_start_time = 0.0
        lateral_duration = 0.0
        bypass_start_time = 0.0
        bypass_duration = 0.0
        bypass_distance_est_m = 0.0
        lateral_effort = 0.0
        avoidance_mode = "none"
        up_start_time = 0.0
        up_duration = 0.0
        vertical_effort = 0.0
        recenter_target_sec = RECENTER_SEC
        last_motion_update_time = time.time()
        reset_smoothing()

    def finalize_attempt(notes_suffix=""):
        nonlocal attempt_active, state, bypass_duration
        nonlocal awaiting_operator_label, pending_label_attempt_id, pending_label_auto_success
        nonlocal pending_label_triggered, pending_label_detection_success, pending_label_notes
        if not attempt_active:
            return
        if bypass_start_time > 0.0:
            bypass_duration = max(bypass_duration, time.time() - bypass_start_time)
        notes = attempt_notes
        if notes_suffix:
            notes = f"{notes} | {notes_suffix}" if notes else notes_suffix
        attempts_writer.writerow([
            attempt_id, scenario_name, PROFILE_NAME, attempt_start_ts, now_iso(),
            battery_start, safe_call(tello.get_battery),
            height_start, safe_call(tello.get_height),
            expected_obstacle_class,
            int(detection_success), int(triggered), int(avoidance_success),
            int(false_trigger), first_detection_frame, trigger_frame, clearance_frame,
            f"{trigger_area:.6f}", f"{trigger_growth:.6f}", f"{trigger_growth_rate:.6f}",
            f"{trigger_ttc_proxy:.3f}", f"{trigger_cx:.6f}",
            f"{trigger_distance_est:.3f}", avoid_direction_text, avoidance_mode,
            f"{up_duration:.3f}", f"{vertical_effort:.3f}",
            f"{lateral_duration:.3f}", f"{lateral_effort:.3f}",
            f"{bypass_duration:.3f}", f"{bypass_distance_est_m:.3f}",
            f"{recenter_target_sec:.3f}", notes
        ])
        attempts_f.flush()
        awaiting_operator_label = True
        pending_label_attempt_id = attempt_id
        pending_label_auto_success = bool(avoidance_success)
        pending_label_triggered = bool(triggered)
        pending_label_detection_success = bool(detection_success)
        pending_label_notes = notes
        attempt_active = False
        state = "HOVER" if flying else "IDLE"
        stop_and_reset()
        log_event(frame_idx, "attempt_finalized", f"id={attempt_id}, success={avoidance_success}, notes={notes}")
        log_event(frame_idx, "operator_label_required", "Press Y=success, F=failure, U=unclear/invalid before next attempt")

    def start_attempt():
        nonlocal attempt_id, attempt_active, attempt_start_ts, attempt_start_time
        nonlocal battery_start, height_start, state, state_start_time, last_motion_update_time
        nonlocal awaiting_operator_label
        if awaiting_operator_label:
            log_event(frame_idx, "start_blocked", "label_previous_attempt_first_Y_F_U")
            return
        if not flying or attempt_active:
            return
        attempt_id += 1
        attempt_start_ts = now_iso()
        attempt_start_time = time.time()
        battery_start = safe_call(tello.get_battery)
        height_start = safe_call(tello.get_height)
        reset_attempt_metrics()
        attempt_active = True
        state = "APPROACH_FORWARD"
        state_start_time = time.time()
        last_motion_update_time = time.time()
        log_event(frame_idx, "attempt_started", f"id={attempt_id}, scenario={scenario_name}, profile={PROFILE_NAME}")

    def label_last_attempt(operator_label, operator_success, note=""):
        nonlocal awaiting_operator_label, labelled_trials, labelled_successes
        if not awaiting_operator_label:
            return
        labels_writer.writerow([
            now_iso(), pending_label_attempt_id, scenario_name, PROFILE_NAME,
            operator_label, int(operator_success) if operator_success is not None else -1,
            int(pending_label_auto_success), int(pending_label_triggered),
            int(pending_label_detection_success),
            f"{pending_label_notes} | {note}" if note else pending_label_notes
        ])
        labels_f.flush()
        labelled_trials += 1
        if operator_success is True:
            labelled_successes += 1
        awaiting_operator_label = False
        log_event(frame_idx, "operator_label_saved", f"attempt={pending_label_attempt_id}, label={operator_label}")

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                continue
            frame = frame.copy()
            raw_frame = frame.copy()
            H, W = frame.shape[:2]
            timestamp = now_iso()
            now_t = time.time()

            if ann_writer is None and SAVE_ANNOTATED_VIDEO:
                ann_writer = cv2.VideoWriter(
                    str(run_dir / "video_annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS,
                    (W, H)
                )
            if raw_writer is None and SAVE_RAW_VIDEO:
                raw_writer = cv2.VideoWriter(
                    str(run_dir / "video_raw.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS,
                    (W, H)
                )


            # Perception
            results = model.predict(source=frame, conf=CONF_THRES, verbose=False)[0]
            detections = []
            primary = None

            if results.boxes is not None and len(results.boxes) > 0:
                for det_idx, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bw = max(0, x2 - x1)
                    bh = max(0, y2 - y1)
                    area_norm = (bw * bh) / float(W * H)
                    cx = ((x1 + x2) / 2.0) / W
                    cy = ((y1 + y2) / 2.0) / H
                    x1n = x1 / W
                    x2n = x2 / W
                    y1n = y1 / H
                    y2n = y2 / H
                    distance_est_det = estimate_distance_from_area(area_norm)
                    is_target = cls_name in TARGET_CLASSES
                    detections.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "x1_norm": x1n, "x2_norm": x2n,
                        "y1_norm": y1n, "y2_norm": y2n,
                        "bbox_area_norm": area_norm,
                        "center_x_norm": cx,
                        "center_y_norm": cy,
                        "distance_est_m": distance_est_det,
                        "is_target": is_target,
                    })

                target_dets = [d for d in detections if d["is_target"]]
                if target_dets:
                    primary = max(target_dets, key=lambda d: d["bbox_area_norm"])


            # Decision
            current_area = 0.0
            current_growth = 0.0
            current_growth_rate = 0.0
            current_cx = 0.5
            distance_est = None
            ttc_proxy = None
            danger = False
            warning = False
            hard_stop = False
            centered = False
            corridor_overlap = False

            if primary is not None:
                current_area = primary["bbox_area_norm"]
                current_cx = primary["center_x_norm"]
                distance_est = primary["distance_est_m"]
                area_history.append((now_t, current_area))
                if len(area_history) >= 2:
                    current_growth = area_history[-1][1] - area_history[0][1]
                    current_growth_rate = compute_area_growth_rate(area_history)
                    ttc_proxy = compute_ttc_proxy(current_area, current_growth_rate)

                centered = CENTER_X_MIN <= current_cx <= CENTER_X_MAX
                corridor_overlap = bbox_intersects_safety_corridor(primary)
                warning = corridor_overlap and current_area >= AREA_WARNING

                area_hard_stop = corridor_overlap and current_area >= AREA_HARD_STOP
                distance_hard_stop = (
                    USE_DISTANCE_TRIGGER and
                    distance_est is not None and
                    distance_est <= HARD_STOP_DISTANCE_M and
                    corridor_overlap
                )
                hard_stop = area_hard_stop or distance_hard_stop

                area_danger_candidate = corridor_overlap and (
                    current_area >= AREA_TRIGGER or
                    (current_area >= AREA_WARNING and current_growth >= GROWTH_TRIGGER)
                )
                distance_danger_candidate = (
                    USE_DISTANCE_TRIGGER and
                    distance_est is not None and
                    distance_est <= TRIGGER_DISTANCE_M and
                    corridor_overlap
                )
                ttc_danger_candidate = (
                    USE_TTC_TRIGGER and
                    ttc_proxy is not None and
                    current_area >= MIN_TTC_AREA and
                    current_growth_rate >= MIN_AREA_GROWTH_RATE and
                    ttc_proxy <= TTC_TRIGGER_SEC and
                    corridor_overlap
                )
                up_danger_candidate = (
                    ENABLE_UPWARD_AVOIDANCE and
                    primary["class_name"] in UP_ONLY_FOR_CLASSES and
                    corridor_overlap and
                    (
                        current_area >= UP_AREA_TRIGGER or
                        (distance_est is not None and distance_est <= UP_DISTANCE_TRIGGER_M)
                    )
                )
                down_danger_candidate = (
                    ENABLE_DOWNWARD_AVOIDANCE and
                    primary["class_name"] in DOWN_ONLY_FOR_CLASSES and
                    corridor_overlap and
                    (
                        current_area >= DOWN_AREA_TRIGGER or
                        (distance_est is not None and distance_est <= DOWN_DISTANCE_TRIGGER_M)
                    )
                )
                danger_candidate = (
                    area_danger_candidate or
                    distance_danger_candidate or
                    ttc_danger_candidate or
                    up_danger_candidate or
                    down_danger_candidate
                )
                confirm_counter = confirm_counter + 1 if danger_candidate else 0
                danger = confirm_counter >= CONFIRM_FRAMES_REQUIRED

                if attempt_active and first_detection_frame < 0 and primary["class_name"] == expected_obstacle_class:
                    first_detection_frame = frame_idx
                    detection_success = True
                    log_event(frame_idx, "first_detection", f"attempt={attempt_id}, class={primary['class_name']}")
            else:
                area_history.append((now_t, 0.0))
                confirm_counter = 0


            for det_i, d in enumerate(detections):
                if not d["is_target"] and not SHOW_NON_TARGETS:
                    continue
                is_primary = primary is not None and d is primary
                intersects = bbox_intersects_safety_corridor(d)
                color = (0, 255, 0) if d["is_target"] else (160, 160, 160)
                if is_primary:
                    color = (0, 0, 255)
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
                dist_txt = ""
                if d["distance_est_m"] is not None:
                    dist_txt = f" d~{d['distance_est_m']:.2f}m"
                label = f"{d['class_name']} {d['confidence']:.2f} A={d['bbox_area_norm']:.3f}{dist_txt}"
                cv2.putText(frame, label, (d["x1"], max(20, d["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
                detections_writer.writerow([
                    timestamp, frame_idx, attempt_id if attempt_active else -1, scenario_name, PROFILE_NAME, state,
                    det_i, d["class_name"], f"{d['confidence']:.4f}",
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    f"{d['x1_norm']:.6f}", f"{d['x2_norm']:.6f}",
                    f"{d['bbox_area_norm']:.6f}", f"{d['center_x_norm']:.6f}", f"{d['center_y_norm']:.6f}",
                    f"{d['distance_est_m']:.3f}" if d["distance_est_m"] is not None else "",
                    int(d["is_target"]), int(is_primary), int(intersects),
                    f"{d.get('y1_norm', 0.0):.6f}", f"{d.get('y2_norm', 0.0):.6f}",
                    f"{bbox_height_norm(d):.6f}", int(obstacle_below_vertical_clearance(d)),
                    int(obstacle_above_vertical_clearance(d))
                ])
            detections_f.flush()


            # Actuation
            if attempt_active and flying:
                if (time.time() - attempt_start_time > MAX_TRIGGER_WAIT_SEC) and not triggered:
                    log_event(frame_idx, "attempt_timeout", f"id={attempt_id}, no_trigger")
                    finalize_attempt("timeout_before_trigger")

                if attempt_active and state == "APPROACH_FORWARD":
                    approach_elapsed = time.time() - state_start_time

                    if hard_stop:
                        send_smooth_rc(EMERGENCY_BACK_SPEED, 0)

                    elif danger and primary is not None and approach_elapsed >= MIN_APPROACH_SEC:
                        triggered = True
                        trigger_frame = frame_idx
                        trigger_area = current_area
                        trigger_growth = current_growth
                        trigger_growth_rate = current_growth_rate
                        trigger_ttc_proxy = ttc_proxy if ttc_proxy is not None else -1.0
                        trigger_cx = current_cx
                        trigger_distance_est = distance_est if distance_est is not None else -1.0
                        current_height_cm = safe_call(tello.get_height)
                        avoidance_mode, avoid_direction_sign, avoid_direction_text, strategy_reason = choose_avoidance_strategy(
                            primary, current_area, distance_est, current_height_cm
                        )
                        stop_and_reset()
                        set_state(
                            "BRAKE_BEFORE_AVOID",
                            f"area={current_area:.4f}, growth={current_growth:.4f}, "
                            f"growth_rate={current_growth_rate:.4f}, ttc={trigger_ttc_proxy:.2f}, "
                            f"dist={trigger_distance_est:.2f}, mode={avoidance_mode}, "
                            f"dir={avoid_direction_text}, {strategy_reason}"
                        )
                    else:
                        send_smooth_rc(RC_APPROACH_FORWARD, 0)

                elif attempt_active and state == "BRAKE_BEFORE_AVOID":
                    stop_and_reset()
                    if time.time() - state_start_time >= BRAKE_BEFORE_AVOID_SEC:
                        exit_counter = 0
                        lost_counter = 0
                        if avoidance_mode == "up":
                            up_start_time = time.time()
                            set_state("AVOID_UP_UNTIL_CLEAR", f"dir=up, target_sec={UP_AVOID_TARGET_SEC:.2f}")
                        elif avoidance_mode == "down":
                            up_start_time = time.time()
                            set_state("AVOID_DOWN_UNTIL_CLEAR", f"dir=down, target_sec={DOWN_AVOID_TARGET_SEC:.2f}")
                        else:
                            lateral_start_time = time.time()
                            set_state("AVOID_LATERAL_UNTIL_EXIT", f"dir={avoid_direction_text}")

                elif attempt_active and state == "AVOID_UP_UNTIL_CLEAR":
                    elapsed = time.time() - up_start_time

                    if primary is not None and obstacle_below_vertical_clearance(primary):
                        exit_counter += 1
                    else:
                        exit_counter = 0

                    if primary is None:
                        lost_counter += 1
                    else:
                        lost_counter = 0

                    up_clear_confirmed = elapsed >= UP_AVOID_MIN_SEC and exit_counter >= EXIT_CORRIDOR_FRAMES_REQUIRED
                    lost_confirmed = elapsed >= UP_AVOID_MIN_SEC and lost_counter >= LOST_CLEAR_FRAMES_REQUIRED
                    target_lift_completed = elapsed >= UP_AVOID_TARGET_SEC

                    if up_clear_confirmed or lost_confirmed or target_lift_completed:
                        stop_and_reset()
                        clearance_frame = frame_idx
                        up_duration = elapsed
                        recenter_target_sec = DOWN_RECENTER_SEC if USE_DOWN_RECENTER_AFTER_UP else 0.0
                        bypass_distance_est_m = 0.0
                        bypass_start_time = time.time()
                        if up_clear_confirmed:
                            reason = "bbox_below_vertical_clearance"
                        elif lost_confirmed:
                            reason = "object_lost_after_vertical_motion"
                        else:
                            reason = "target_lift_time_completed"
                        set_state(
                            "FORWARD_AROUND_OBSTACLE",
                            f"vertical_{reason}, up_sec={elapsed:.2f}, vertical_effort={vertical_effort:.1f}"
                        )
                    elif elapsed >= UP_AVOID_MAX_SEC:
                        stop_and_reset()
                        up_duration = elapsed
                        lateral_start_time = time.time()
                        set_state("AVOID_LATERAL_UNTIL_EXIT", "vertical_max_time_reached_fallback_to_lateral")
                    else:
                        motion_now = time.time()
                        dt_motion = min(max(motion_now - last_motion_update_time, 0.0), 0.20)
                        last_motion_update_time = motion_now
                        vertical_effort += abs(RC_UP_AVOID) * dt_motion
                        send_smooth_rc(0, 0, RC_UP_AVOID)

                elif attempt_active and state == "AVOID_DOWN_UNTIL_CLEAR":
                    elapsed = time.time() - up_start_time

                    if primary is not None and obstacle_above_vertical_clearance(primary):
                        exit_counter += 1
                    else:
                        exit_counter = 0

                    if primary is None:
                        lost_counter += 1
                    else:
                        lost_counter = 0

                    down_clear_confirmed = elapsed >= DOWN_AVOID_MIN_SEC and exit_counter >= EXIT_CORRIDOR_FRAMES_REQUIRED
                    lost_confirmed = elapsed >= DOWN_AVOID_MIN_SEC and lost_counter >= LOST_CLEAR_FRAMES_REQUIRED
                    target_drop_completed = elapsed >= DOWN_AVOID_TARGET_SEC

                    if down_clear_confirmed or lost_confirmed or target_drop_completed:
                        stop_and_reset()
                        clearance_frame = frame_idx
                        up_duration = elapsed
                        recenter_target_sec = UP_RECENTER_AFTER_DOWN_SEC if USE_UP_RECENTER_AFTER_DOWN else 0.0
                        bypass_distance_est_m = 0.0
                        bypass_start_time = time.time()
                        if down_clear_confirmed:
                            reason = "bbox_above_downward_clearance"
                        elif lost_confirmed:
                            reason = "object_lost_after_downward_motion"
                        else:
                            reason = "target_drop_time_completed"
                        set_state(
                            "FORWARD_AROUND_OBSTACLE",
                            f"vertical_{reason}, down_sec={elapsed:.2f}, vertical_effort={vertical_effort:.1f}"
                        )
                    elif elapsed >= DOWN_AVOID_MAX_SEC:
                        stop_and_reset()
                        up_duration = elapsed
                        lateral_start_time = time.time()
                        set_state("AVOID_LATERAL_UNTIL_EXIT", "downward_max_time_reached_fallback_to_lateral")
                    else:
                        motion_now = time.time()
                        dt_motion = min(max(motion_now - last_motion_update_time, 0.0), 0.20)
                        last_motion_update_time = motion_now
                        vertical_effort += abs(RC_DOWN_AVOID) * dt_motion
                        send_smooth_rc(0, 0, -RC_DOWN_AVOID)

                elif attempt_active and state == "AVOID_LATERAL_UNTIL_EXIT":
                    elapsed = time.time() - lateral_start_time

                    if primary is not None and bbox_outside_safety_corridor(primary):
                        exit_counter += 1
                    else:
                        exit_counter = 0

                    if primary is None:
                        lost_counter += 1
                    else:
                        lost_counter = 0

                    exit_confirmed = elapsed >= MIN_LATERAL_SEC and exit_counter >= EXIT_CORRIDOR_FRAMES_REQUIRED
                    lost_confirmed = elapsed >= MIN_LATERAL_SEC and lost_counter >= LOST_CLEAR_FRAMES_REQUIRED

                    if exit_confirmed or lost_confirmed:
                        stop_and_reset()
                        clearance_frame = frame_idx
                        lateral_duration = elapsed

                        if USE_EFFORT_BASED_RECENTER:
                            recenter_target_sec = (lateral_effort / max(RC_RECENTER, 1)) * RECENTER_COMPENSATION
                            recenter_target_sec = max(MIN_RECENTER_SEC, min(MAX_RECENTER_SEC, recenter_target_sec))
                        else:
                            recenter_target_sec = RECENTER_SEC

                        bypass_distance_est_m = 0.0
                        bypass_start_time = time.time()
                        reason = "bbox_outside_safety_corridor" if exit_confirmed else "object_lost_after_multiple_frames"
                        set_state(
                            "FORWARD_AROUND_OBSTACLE",
                            f"{reason}, lateral_sec={elapsed:.2f}, lateral_effort={lateral_effort:.1f}, "
                            f"recenter_target={recenter_target_sec:.2f}"
                        )
                    elif elapsed >= MAX_LATERAL_SEC:
                        stop_and_reset()
                        lateral_duration = elapsed
                        finalize_attempt("lateral_timeout_obstacle_not_outside_safety_corridor")
                    else:
                        if elapsed < 1.20:
                            lateral_speed = RC_LATERAL_BASE
                        elif elapsed < 2.40:
                            lateral_speed = RC_LATERAL_STRONG
                        else:
                            lateral_speed = RC_LATERAL_MAX
                        lr = lateral_speed if avoid_direction_sign == +1 else -lateral_speed

                        motion_now = time.time()
                        dt_motion = min(max(motion_now - last_motion_update_time, 0.0), 0.20)
                        last_motion_update_time = motion_now
                        lateral_effort += abs(lr) * dt_motion

                        send_smooth_rc(0, lr)

                elif attempt_active and state == "FORWARD_AROUND_OBSTACLE":
                    bypass_elapsed = time.time() - bypass_start_time
                    can_recheck = bypass_elapsed >= MIN_BYPASS_BEFORE_RECHECK_SEC
                    vertical_clear_during_bypass = (
                        primary is not None and
                        ((avoidance_mode == "up" and obstacle_below_vertical_clearance(primary)) or
                         (avoidance_mode == "down" and obstacle_above_vertical_clearance(primary)))
                    )
                    obstacle_reentered = (
                        primary is not None and
                        bbox_intersects_safety_corridor(primary) and
                        primary["bbox_area_norm"] > BYPASS_RECHECK_AREA_MIN and
                        not vertical_clear_during_bypass
                    )
                    if can_recheck and obstacle_reentered:
                        exit_counter = 0
                        lost_counter = 0
                        if avoidance_mode == "up":
                            up_start_time = time.time()
                            set_state("AVOID_UP_UNTIL_CLEAR", "obstacle_seen_during_vertical_bypass")
                        elif avoidance_mode == "down":
                            up_start_time = time.time()
                            set_state("AVOID_DOWN_UNTIL_CLEAR", "obstacle_seen_during_vertical_bypass")
                        else:
                            lateral_start_time = time.time()
                            set_state("AVOID_LATERAL_UNTIL_EXIT", "obstacle_reentered_safety_corridor_during_bypass")
                    else:
                        motion_now = time.time()
                        dt_motion = min(max(motion_now - last_motion_update_time, 0.0), 0.20)
                        last_motion_update_time = motion_now

                        lr_cmd = 0
                        if BYPASS_MODE == "diagonal" and bypass_elapsed <= BYPASS_DIAGONAL_SEC:
                            lr_cmd = BYPASS_DIAGONAL_LR if avoid_direction_sign == +1 else -BYPASS_DIAGONAL_LR
                            lateral_effort += abs(lr_cmd) * dt_motion

                        forward_mps = estimate_forward_mps(RC_BYPASS_FORWARD)
                        bypass_distance_est_m += forward_mps * dt_motion

                        send_smooth_rc(RC_BYPASS_FORWARD, lr_cmd)

                        if USE_DISTANCE_BASED_BYPASS:
                            bypass_completed = (
                                bypass_distance_est_m >= BYPASS_TARGET_DISTANCE_M
                                and bypass_elapsed >= MIN_BYPASS_FORWARD_SEC
                            ) or bypass_elapsed >= MAX_BYPASS_FORWARD_SEC
                        else:
                            bypass_completed = bypass_elapsed >= BYPASS_FORWARD_SEC

                        if bypass_completed:
                            bypass_duration = bypass_elapsed
                            recenter_clear_counter = 0
                            stop_and_reset()
                            set_state(
                                "RECENTER_AFTER_PASS",
                                f"around_forward_completed, bypass_dist_est={bypass_distance_est_m:.2f}m, "
                                f"bypass_sec={bypass_duration:.2f}"
                            )

                elif attempt_active and state == "RECENTER_AFTER_PASS":
                    obstacle_seen = (
                        primary is not None and
                        bbox_intersects_safety_corridor(primary) and
                        primary["bbox_area_norm"] > BYPASS_RECHECK_AREA_MIN
                    )
                    if avoidance_mode == "up" and not USE_DOWN_RECENTER_AFTER_UP:
                        stop_and_reset()
                        set_state("FINAL_FORWARD", "vertical_bypass_no_down_recenter")
                    elif avoidance_mode == "up" and USE_DOWN_RECENTER_AFTER_UP:
                        send_smooth_rc(0, 0, -RC_DOWN_RECENTER)
                        if time.time() - state_start_time >= recenter_target_sec:
                            stop_and_reset()
                            set_state("FINAL_FORWARD", "down_recenter_completed")
                    elif avoidance_mode == "down" and not USE_UP_RECENTER_AFTER_DOWN:
                        stop_and_reset()
                        set_state("FINAL_FORWARD", "vertical_bypass_no_up_recenter")
                    elif avoidance_mode == "down" and USE_UP_RECENTER_AFTER_DOWN:
                        send_smooth_rc(0, 0, RC_UP_RECENTER_AFTER_DOWN)
                        if time.time() - state_start_time >= recenter_target_sec:
                            stop_and_reset()
                            set_state("FINAL_FORWARD", "up_recenter_after_down_completed")
                    elif obstacle_seen:
                        lateral_start_time = time.time()
                        exit_counter = 0
                        lost_counter = 0
                        set_state("AVOID_LATERAL_UNTIL_EXIT", "obstacle_seen_during_recenter")
                    else:
                        lr = -RC_RECENTER if avoid_direction_sign == +1 else RC_RECENTER
                        send_smooth_rc(6, lr)
                        if time.time() - state_start_time >= recenter_target_sec:
                            stop_and_reset()
                            set_state("FINAL_FORWARD", "recenter_completed")

                elif attempt_active and state == "FINAL_FORWARD":
                    obstacle_seen = (
                        primary is not None and
                        bbox_intersects_safety_corridor(primary) and
                        primary["bbox_area_norm"] > BYPASS_RECHECK_AREA_MIN and
                        not ((avoidance_mode == "up" and obstacle_below_vertical_clearance(primary)) or
                             (avoidance_mode == "down" and obstacle_above_vertical_clearance(primary)))
                    )
                    if obstacle_seen:
                        exit_counter = 0
                        lost_counter = 0
                        if avoidance_mode == "up":
                            up_start_time = time.time()
                            set_state("AVOID_UP_UNTIL_CLEAR", "obstacle_seen_during_final_forward")
                        elif avoidance_mode == "down":
                            up_start_time = time.time()
                            set_state("AVOID_DOWN_UNTIL_CLEAR", "obstacle_seen_during_final_forward")
                        else:
                            lateral_start_time = time.time()
                            set_state("AVOID_LATERAL_UNTIL_EXIT", "obstacle_seen_during_final_forward")
                    else:
                        send_smooth_rc(RC_FINAL_FORWARD, 0)
                        if time.time() - state_start_time >= FINAL_FORWARD_SEC:
                            stop_and_reset()
                            avoidance_success = True
                            finalize_attempt("avoidance_completed_v6_distance_bypass_effort_recenter")
            else:
                if flying:
                    stop_and_reset()


            center_min_px = int(CENTER_X_MIN * W)
            center_max_px = int(CENTER_X_MAX * W)
            safety_min_px = int(SAFETY_X_MIN * W)
            safety_max_px = int(SAFETY_X_MAX * W)
            cv2.rectangle(frame, (safety_min_px, 0), (safety_max_px, H), (255, 255, 0), 2)
            cv2.rectangle(frame, (center_min_px, 0), (center_max_px, H), (255, 180, 0), 1)
            if ENABLE_UPWARD_AVOIDANCE:
                min_top_y_px = int(UP_MIN_TOP_CLEARANCE * H)
                head_zone_y_px = int(UP_HEAD_Y1_MAX * H)
                head_center_y_px = int(UP_HEAD_CENTER_Y_MAX * H)
                clear_top_y_px = int(UP_CLEAR_TOP_Y_NORM * H)
                cv2.line(frame, (0, min_top_y_px), (W, min_top_y_px), (180, 180, 255), 1)
                cv2.line(frame, (0, head_zone_y_px), (W, head_zone_y_px), (0, 180, 255), 2)
                cv2.line(frame, (0, head_center_y_px), (W, head_center_y_px), (0, 140, 255), 1)
                cv2.line(frame, (0, clear_top_y_px), (W, clear_top_y_px), (0, 255, 255), 1)
                cv2.putText(frame, "UP head y1 zone", (12, max(18, head_zone_y_px - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)
                cv2.putText(frame, "UP head cy limit", (12, max(18, head_center_y_px - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1)
                cv2.putText(frame, "UP clear line", (12, max(18, clear_top_y_px - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            if ENABLE_DOWNWARD_AVOIDANCE:
                min_bottom_y_px = int((1.0 - DOWN_MIN_BOTTOM_CLEARANCE) * H)
                clear_bottom_y_px = int(DOWN_CLEAR_BOTTOM_Y_NORM * H)
                cv2.line(frame, (0, min_bottom_y_px), (W, min_bottom_y_px), (255, 180, 180), 1)
                cv2.line(frame, (0, clear_bottom_y_px), (W, clear_bottom_y_px), (0, 180, 255), 1)
                cv2.putText(frame, "DOWN min bottom clearance", (12, min(H - 8, min_bottom_y_px + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 180), 1)
                cv2.putText(frame, "DOWN clear line", (12, min(H - 8, clear_bottom_y_px + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)
            draw_overlay(
                frame,
                flying,
                state,
                scenario_name,
                attempt_active,
                primary,
                distance_est,
                ttc_proxy,
                danger,
                warning,
                exit_counter,
                lost_counter,
                lateral_effort,
                recenter_target_sec,
                bypass_distance_est_m,
                avoidance_mode,
                vertical_effort,
                up_duration,
            )
            if awaiting_operator_label:
                cv2.putText(frame, "LABEL REQUIRED: Y=success | F=failure | U=unclear", (12, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Labelled trials: {labelled_trials} | Operator successes: {labelled_successes}", (12, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if SAVE_PERIODIC_FRAMES and (time.time() - last_frame_save >= PERIODIC_FRAME_INTERVAL_SEC):
                cv2.imwrite(str(run_dir / "frames_raw" / f"frame_{frame_idx:06d}_raw.jpg"), raw_frame)
                cv2.imwrite(str(run_dir / "frames_annotated" / f"frame_{frame_idx:06d}_ann.jpg"), frame)
                last_frame_save = time.time()

            if time.time() - last_telemetry_log >= TELEMETRY_LOG_INTERVAL_SEC:
                telemetry_writer.writerow([
                    timestamp, frame_idx, attempt_id if attempt_active else -1, scenario_name, PROFILE_NAME,
                    safe_call(tello.get_battery), safe_call(tello.get_height), safe_call(tello.get_distance_tof),
                    safe_call(tello.get_yaw), safe_call(tello.get_pitch), safe_call(tello.get_roll),
                    safe_call(tello.get_speed_x), safe_call(tello.get_speed_y), safe_call(tello.get_speed_z),
                    state, int(flying), int(attempt_active),
                    f"{lateral_effort:.3f}", f"{bypass_distance_est_m:.3f}", f"{recenter_target_sec:.3f}",
                    avoidance_mode, f"{vertical_effort:.3f}", f"{up_duration:.3f}"
                ])
                telemetry_f.flush()
                last_telemetry_log = time.time()

            if ann_writer is not None:
                ann_writer.write(frame)
            if raw_writer is not None:
                raw_writer.write(raw_frame)

            cv2.imshow(WINDOW_NAME, cv2.resize(frame, (960, 540)))
            # Manual interface
            key = cv2.waitKeyEx(1)


            if key in (27, ord('q'), ord('Q')):
                if attempt_active:
                    finalize_attempt("quit_during_attempt")
                break
            elif key in (ord('t'), ord('T')):
                if not flying:
                    try:
                        tello.takeoff()
                        time.sleep(TAKEOFF_STABILIZE_SEC)
                        stop_and_reset()
                        flying = True
                        state = "HOVER"
                        log_event(frame_idx, "takeoff_ok", f"battery={safe_call(tello.get_battery)}")
                    except Exception as e:
                        log_event(frame_idx, "takeoff_error", str(e))
                        stop_and_reset()
            elif key in (ord('l'), ord('L')):
                if attempt_active:
                    finalize_attempt("manual_land_during_attempt")
                if flying:
                    stop_and_reset()
                    try:
                        tello.land()
                    except Exception as e:
                        log_event(frame_idx, "land_error", str(e))
                    flying = False
                    state = "IDLE"
            elif key == ord('1'):
                scenario_name = "centered_person"
                expected_obstacle_class = "person"
            elif key == ord('2'):
                scenario_name = "left_person"
                expected_obstacle_class = "person"
            elif key == ord('3'):
                scenario_name = "right_person"
                expected_obstacle_class = "person"
            elif key == ord('4'):
                scenario_name = "centered_car"
                expected_obstacle_class = "car"
            elif key == ord('5'):
                scenario_name = "centered_person_upward_escape"
                expected_obstacle_class = "person"
            elif key == ord('6'):
                scenario_name = "centered_person_downward_escape"
                expected_obstacle_class = "person"
            elif key in (ord('y'), ord('Y')):
                label_last_attempt("success", True)
            elif key in (ord('f'), ord('F')):
                label_last_attempt("failure", False)
            elif key in (ord('u'), ord('U')):
                label_last_attempt("unclear", None, "invalid_or_unclear_trial")
            elif key == 13:
                start_attempt()
            elif key in (ord('x'), ord('X')):
                if attempt_active:
                    finalize_attempt("manual_abort")
            elif key == ARROW_UP and flying and not attempt_active:
                tello.send_rc_control(0, RC_MANUAL_SPEED, 0, 0)
            elif key == ARROW_DOWN and flying and not attempt_active:
                tello.send_rc_control(0, -RC_MANUAL_SPEED, 0, 0)
            elif key == ARROW_LEFT and flying and not attempt_active:
                tello.send_rc_control(-RC_MANUAL_SPEED, 0, 0, 0)
            elif key == ARROW_RIGHT and flying and not attempt_active:
                tello.send_rc_control(RC_MANUAL_SPEED, 0, 0, 0)
            elif key in (ord('w'), ord('W')) and flying and not attempt_active:
                tello.send_rc_control(0, 0, RC_UPDOWN_SPEED, 0)
            elif key in (ord('s'), ord('S')) and flying and not attempt_active:
                tello.send_rc_control(0, 0, -RC_UPDOWN_SPEED, 0)
            elif key in (ord('a'), ord('A')) and flying and not attempt_active:
                tello.send_rc_control(0, 0, 0, -RC_YAW_SPEED)
            elif key in (ord('d'), ord('D')) and flying and not attempt_active:
                tello.send_rc_control(0, 0, 0, RC_YAW_SPEED)
            elif not attempt_active and key == -1:
                if flying:
                    stop_and_reset()

            frame_idx += 1

    except Exception as e:
        traceback.print_exc()
        try:
            log_event(frame_idx, "exception", str(e))
        except Exception:
            pass
    finally:
        # Cleanup
        try:
            if attempt_active:
                finalize_attempt("cleanup_finalize")
        except Exception:
            pass
        try:
            stop_and_reset()
        except Exception:
            pass
        try:
            if flying:
                tello.land()
        except Exception as e:
            try:
                log_event(frame_idx, "cleanup_land_error", str(e))
            except Exception:
                pass
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass
        for fobj in (events_f, attempts_f, detections_f, telemetry_f, labels_f):
            try:
                fobj.close()
            except Exception:
                pass
        if ann_writer is not None:
            ann_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
