"""
record.py - Multi-modal tactile data collection script for Daimon sensor.

Usage:
    python record.py

Controls (in the OpenCV window):
    SPACE  - Start recording 30 frames for current trial
    r      - Reset sensor baseline (do this BEFORE each trial)
    n      - Next material (you'll be prompted to enter material name)
    q      - Quit and disconnect sensor

Data is saved to: dataset/material=<name>/trial_<NNN>/
"""

import os
import sys
import time
import json
import select
import termios
import tty
import atexit
import cv2
import numpy as np
from datetime import datetime
from dmrobotics import Sensor, put_arrows_on_image


def print_event(msg):
    """Print msg on its own line, fully clearing any in-place FPS line first."""
    sys.stdout.write("\r\033[2K" + msg + "\n")
    sys.stdout.flush()


class TerminalKeyReader:
    """Non-blocking single-keystroke reader from stdin (cbreak mode).

    Lets the user control the script from the terminal even when the OpenCV
    windows don't have focus. Falls back to a no-op if stdin is not a TTY.
    """

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._old_attrs = None
        if self.enabled:
            self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            atexit.register(self.restore)

    def restore(self):
        if self.enabled and self._old_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def poll(self):
        """Return the most recent pending byte (drains the queue) or None."""
        if not self.enabled:
            return None
        last = None
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            ch = os.read(sys.stdin.fileno(), 1)
            if not ch:
                break
            last = ch[0]
        return last


def drain_keys(term_reader, wait_ms=15):
    """Combined key source: OpenCV waitKey + terminal stdin.

    Drains ALL queued keys from both sources, returning the most recent
    meaningful one (or 255 if none). Works regardless of which window
    currently has focus.
    """
    k = cv2.waitKey(wait_ms) & 0xFF
    last = k
    while True:
        k2 = cv2.waitKey(1) & 0xFF
        if k2 == 255:
            break
        last = k2
    term_k = term_reader.poll()
    if term_k is not None:
        last = term_k
    return last


# ============================================================
# CONFIGURATION
# ============================================================
DEV_SERIAL_ID = "S2508080042"   # Change if your sensor differs
FRAMES_PER_TRIAL = 30            # ~1 second at 30 FPS
DATASET_ROOT = "dataset"         # Where data is saved


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_next_trial_index(material_dir):
    """Find the next available trial number for this material."""
    if not os.path.exists(material_dir):
        return 1
    existing = [d for d in os.listdir(material_dir) if d.startswith("trial_")]
    if not existing:
        return 1
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    return max(nums) + 1 if nums else 1


def record_trial(sensor, material, trial_idx, black_img):
    """Record FRAMES_PER_TRIAL frames of all 4 modalities and save to disk."""
    
    # Prepare output directory
    material_dir = os.path.join(DATASET_ROOT, f"material={material}")
    trial_dir = os.path.join(material_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Buffers for the 4 modalities
    raw_imgs = []
    depths = []
    deformations = []
    shears = []
    timestamps = []
    
    print_event(f"[REC] Recording trial {trial_idx} for material '{material}'...")
    start = time.time()
    
    for i in range(FRAMES_PER_TRIAL):
        img = sensor.getRawImage()
        depth = sensor.getDepth()
        deformation = sensor.getDeformation2D()
        shear = sensor.getShear()
        
        raw_imgs.append(img.copy())
        depths.append(depth.copy())
        deformations.append(deformation.copy())
        shears.append(shear.copy())
        timestamps.append(time.time())
        
        # Live preview during recording (so user knows it's working)
        depth_img = cv2.applyColorMap(
            (depth * 0.25 * 255.0).astype('uint8'), cv2.COLORMAP_HOT
        )
        cv2.imshow('img', img)
        cv2.imshow('depth', depth_img)
        cv2.imshow('deformation', put_arrows_on_image(black_img, deformation * 20))
        cv2.imshow('shear', put_arrows_on_image(black_img, shear * 20))
        cv2.waitKey(1)
    
    duration = time.time() - start
    
    # Save all four modalities as .npy files
    np.save(os.path.join(trial_dir, "rawimg.npy"), np.array(raw_imgs))
    np.save(os.path.join(trial_dir, "depth.npy"), np.array(depths))
    np.save(os.path.join(trial_dir, "deformation.npy"), np.array(deformations))
    np.save(os.path.join(trial_dir, "shear.npy"), np.array(shears))
    
    # Save metadata
    meta = {
        "material": material,
        "trial": trial_idx,
        "timestamp": datetime.now().isoformat(),
        "frames": FRAMES_PER_TRIAL,
        "duration_sec": round(duration, 3),
        "sensor_id": DEV_SERIAL_ID,
        "frame_timestamps": timestamps,
    }
    with open(os.path.join(trial_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print_event(f"[OK]  Saved {FRAMES_PER_TRIAL} frames to {trial_dir} "
                f"(duration: {duration:.2f}s)")


def prompt_material(window_name="prompt"):
    """Ask the user for the current material name via an OpenCV window.

    Reading from terminal stdin while OpenCV windows are open is unreliable —
    keystrokes go to whichever window has focus, and OpenCV swallows them.
    Capture characters through cv2.waitKey instead so focus stays in the GUI.

    Enter to submit, Backspace to delete, ESC to keep current value.
    """
    print("\n" + "=" * 50)
    print("Type material name in the 'prompt' window. Enter=OK, ESC=cancel.")
    print("=" * 50)

    buf = ""
    canvas_h, canvas_w = 120, 600
    while True:
        img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(img, "Material name (Enter=OK, Bksp=del, ESC=cancel):",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(img, buf + "_", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(window_name, img)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        k = cv2.waitKey(20) & 0xFF

        if k == 255:                      # no key
            continue
        if k in (13, 10):                 # Enter
            name = buf.strip()
            if name:
                break
        elif k == 27:                     # ESC -> cancel
            cv2.destroyWindow(window_name)
            return None
        elif k in (8, 127):               # Backspace / Delete
            buf = buf[:-1]
        elif 32 <= k <= 126:              # printable ASCII
            buf += chr(k)

    cv2.destroyWindow(window_name)
    name = name.replace(" ", "_").lower()
    print(f"Now collecting for material: '{name}'")
    return name


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    # Terminal in cbreak so single keystrokes from stdin work too
    term_reader = TerminalKeyReader()

    # Connect to sensor
    print(f"Connecting to Daimon sensor (serial: {DEV_SERIAL_ID})...")
    sensor = Sensor(DEV_SERIAL_ID)
    print("Sensor connected.")
    
    # Setup
    black_img = np.zeros_like(sensor.getRawImage())
    black_img = np.stack([black_img] * 3, axis=-1)
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    # Initial material selection (loop until non-empty — ESC has no fallback here)
    current_material = None
    while not current_material:
        current_material = prompt_material()
    trial_idx = get_next_trial_index(
        os.path.join(DATASET_ROOT, f"material={current_material}")
    )
    
    print("\n" + "=" * 50)
    print("CONTROLS:")
    print("  SPACE - record one trial (30 frames)")
    print("  r     - reset sensor baseline (do BEFORE each trial)")
    print("  n     - switch to a new material")
    print("  q     - quit and save")
    print("=" * 50)
    print(f"\nReady. Next trial: {trial_idx} ({current_material})")
    
    frame_count = 0
    fps_start = time.time()
    
    try:
        while True:
            # Live preview loop
            img = sensor.getRawImage()
            depth = sensor.getDepth()
            deformation = sensor.getDeformation2D()
            shear = sensor.getShear()
            
            depth_img = cv2.applyColorMap(
                (depth * 0.25 * 255.0).astype('uint8'), cv2.COLORMAP_HOT
            )
            cv2.imshow('img', img)
            cv2.imshow('depth', depth_img)
            cv2.imshow('deformation', put_arrows_on_image(black_img, deformation * 20))
            cv2.imshow('shear', put_arrows_on_image(black_img, shear * 20))
            
            frame_count += 1
            
            # FPS counter (\r overwrite — pad with spaces so it fully clears)
            if time.time() - fps_start > 1.0:
                fps = frame_count / (time.time() - fps_start)
                line = (f"  [Live FPS: {fps:.1f}  |  Material: {current_material}  "
                        f"|  Next trial: {trial_idx}]")
                sys.stdout.write("\r\033[2K" + line)
                sys.stdout.flush()
                frame_count = 0
                fps_start = time.time()

            # Drain queued keys from both OpenCV and terminal stdin
            k = drain_keys(term_reader, wait_ms=15)

            if k == ord('q'):
                print_event("Quitting...")
                break

            elif k == ord('r'):
                sensor.reset()
                print_event(f"[RESET] Sensor baseline zeroed. Ready for trial {trial_idx}.")

            elif k == ord('n'):
                new_material = prompt_material()
                if new_material:                       # ESC -> keep current material
                    current_material = new_material
                    trial_idx = get_next_trial_index(
                        os.path.join(DATASET_ROOT, f"material={current_material}")
                    )
                    print_event(f"Next trial: {trial_idx}")
                else:
                    print_event("Material change cancelled.")

            elif k == ord(' '):  # SPACE = record
                record_trial(sensor, current_material, trial_idx, black_img)
                trial_idx += 1
                print_event(f"Ready for next trial: {trial_idx}")
    
    finally:
        term_reader.restore()
        sensor.disconnect()
        cv2.destroyAllWindows()
        print("Sensor disconnected. Bye!")


if __name__ == "__main__":
    main()
