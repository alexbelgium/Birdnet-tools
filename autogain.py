#!/usr/bin/env python3
"""
https://github.com/alexbelgium/Birdnet-tools/edit/main/autogain.py
Dynamic Microphone Gain Adjustment Script with Interactive Calibration,
Self‑Modification, No‑Signal Reboot Logic, and a Test Mode for Real‑Time RMS Line Graph using plotext

Usage:
  ./autogain.py                 -> Normal dynamic gain control
  ./autogain.py --calibrate     -> Interactive calibration + self-modification
  ./autogain.py --test          -> Test mode (real-time RMS graph)
"""

import argparse
import subprocess
import numpy as np
from scipy.signal import butter, sosfilt
import time
import re
import sys
import os

# ---------------------- Default Configuration ----------------------

MICROPHONE_NAME = "Line In 1 Gain"
MIN_GAIN_DB = 30
MAX_GAIN_DB = 38
GAIN_STEP_DB = 3

# RMS thresholds
NOISE_THRESHOLD_HIGH = 0.01
NOISE_THRESHOLD_LOW  = 0.001

# No-signal detection
NO_SIGNAL_THRESHOLD = 1e-6
NO_SIGNAL_COUNT_THRESHOLD = 3
NO_SIGNAL_ACTION = "scarlett2 reboot && sudo reboot"

SAMPLING_RATE = 48000  # 48 kHz
LOWCUT        = 2000
HIGHCUT       = 8000
FILTER_ORDER  = 4
RTSP_URL      = "rtsp://192.168.178.124:8554/birdmic"
SLEEP_SECONDS = 10

REFERENCE_PRESSURE = 20e-6  # 20 µPa

# Default microphone specifications (for calibration reference)
DEFAULT_SNR         = 80.0    # dB
DEFAULT_SELF_NOISE  = 14.0    # dB-A
DEFAULT_CLIPPING    = 120.0   # dB SPL
DEFAULT_SENSITIVITY = -28.0   # dB re 1 V/Pa

# Compute the default full-scale amplitude (used to derive default fractions)
def_full_scale = (
    REFERENCE_PRESSURE *
    10 ** (DEFAULT_CLIPPING / 20) *
    10 ** (DEFAULT_SENSITIVITY / 20)
)

# ---------------------- Argument Parsing ----------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic Mic Gain Adjustment with calibration, test mode, self‑modification, and reboot logic."
    )
    parser.add_argument("--calibrate", action="store_true", help="Run interactive calibration mode")
    parser.add_argument("--test", action="store_true", help="Run test mode to display a real‑time RMS graph using plotext")
    return parser.parse_args()

# ---------------------- Audio & Gain Helpers ----------------------

def debug_print(msg, level="info"):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] [{level.upper()}] {msg}")

def get_gain_db(mic_name):
    try:
        output = subprocess.check_output(
            ['amixer', 'sget', mic_name], stderr=subprocess.STDOUT
        ).decode()
        match = re.search(r'\[(-?\d+(\.\d+)?)dB\]', output)
        if match:
            return float(match.group(1))
    except subprocess.CalledProcessError as e:
        debug_print(f"amixer sget failed: {e}", "error")
    return None

def set_gain_db(mic_name, gain_db):
    gain_db = max(min(gain_db, MAX_GAIN_DB), MIN_GAIN_DB)
    try:
        subprocess.check_call(
            ['amixer', 'sset', mic_name, f'{int(gain_db)}dB'],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        debug_print(f"Gain set to: {gain_db} dB", "info")
        return True
    except subprocess.CalledProcessError as e:
        debug_print(f"Failed to set gain: {e}", "error")
    return False

def capture_audio(rtsp_url, duration=5):
    cmd = [
        'ffmpeg', '-loglevel', 'error', '-rtsp_transport', 'tcp',
        '-i', rtsp_url, '-vn', '-f', 's16le', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLING_RATE), '-ac', '1', '-t', str(duration), '-'
    ]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            debug_print(f"ffmpeg failed: {stderr.decode().strip()}", "error")
            return None
        return np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        debug_print(f"Audio capture exception: {e}", "error")
        return None

def bandpass_filter(audio, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, audio)

def measure_rms(audio):
    return float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

# ---------------------- Interactive Calibration ----------------------

def prompt_float(prompt_str, default_val):
    while True:
        user_input = input(f"{prompt_str} [{default_val}]: ").strip()
        if user_input == "":
            return default_val
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input; please enter a numeric value.")

def interactive_calibration():
    print("\n-- INTERACTIVE CALIBRATION --")
    print("Enter the microphone characteristics (press Enter to accept default):\n")
    snr = prompt_float("1) Signal-to-Noise Ratio (dB)", DEFAULT_SNR)
    self_noise = prompt_float("2) Self Noise (dB-A)", DEFAULT_SELF_NOISE)
    clipping = prompt_float("3) Clipping SPL (dB)", DEFAULT_CLIPPING)
    sensitivity = prompt_float("4) Sensitivity (dB re 1 V/Pa)", DEFAULT_SENSITIVITY)
    return {"snr": snr, "self_noise": self_noise, "clipping": clipping, "sensitivity": sensitivity}

def calibrate_and_propose(mic_params):
    user_snr = mic_params["snr"]
    clipping = mic_params["clipping"]
    sensitivity = mic_params["sensitivity"]

    user_full_scale = (
        REFERENCE_PRESSURE *
        10 ** (clipping / 20) *
        10 ** (sensitivity / 20)
    )
    fraction_high_default = NOISE_THRESHOLD_HIGH / def_full_scale
    fraction_low_default  = NOISE_THRESHOLD_LOW  / def_full_scale
    snr_ratio = user_snr / DEFAULT_SNR

    proposed_high = fraction_high_default * user_full_scale * snr_ratio
    proposed_low  = fraction_low_default  * user_full_scale * snr_ratio
    gain_offset = (DEFAULT_SENSITIVITY - sensitivity)
    proposed_min_gain = MIN_GAIN_DB + gain_offset
    proposed_max_gain = MAX_GAIN_DB + gain_offset

    print("\n===============================================================")
    print("CURRENT VALUES:")
    print("---------------------------------------------------------------")
    print(f"  NOISE_THRESHOLD_HIGH: {NOISE_THRESHOLD_HIGH:.7f}")
    print(f"  NOISE_THRESHOLD_LOW:  {NOISE_THRESHOLD_LOW:.7f}")
    print(f"  MIN_GAIN_DB:          {MIN_GAIN_DB}")
    print(f"  MAX_GAIN_DB:          {MAX_GAIN_DB}")
    print("---------------------------------------------------------------\n")
    print("PROPOSED VALUES:")
    print("---------------------------------------------------------------")
    print(f"  Proposed NOISE_THRESHOLD_HIGH: {proposed_high:.7f}")
    print(f"  Proposed NOISE_THRESHOLD_LOW:  {proposed_low:.7f}\n")
    print("  Proposed Gain Range (dB):")
    print(f"    MIN_GAIN_DB: {proposed_min_gain:.2f}")
    print(f"    MAX_GAIN_DB: {proposed_max_gain:.2f}")
    print("---------------------------------------------------------------\n")

    return {
        "noise_threshold_high": proposed_high,
        "noise_threshold_low": proposed_low,
        "min_gain_db": proposed_min_gain,
        "max_gain_db": proposed_max_gain,
    }

def persist_calibration_to_script(script_path, proposal):
    subs = {
        "NOISE_THRESHOLD_HIGH": f"{proposal['noise_threshold_high']:.7f}",
        "NOISE_THRESHOLD_LOW":  f"{proposal['noise_threshold_low']:.7f}",
        "MIN_GAIN_DB":          f"{int(round(proposal['min_gain_db']))}",
        "MAX_GAIN_DB":          f"{int(round(proposal['max_gain_db']))}"
    }
    for var, val in subs.items():
        cmd = f"sed -i 's|^{var} = .*|{var} = {val}|' \"{script_path}\""
        os.system(cmd)
    print("✅ Script has been updated with the new calibration values.\n")

# ---------------------- Test Mode: Real-Time RMS Graph using plotext ----------------------

def test_mode():
    try:
        import plotext as plt
    except ImportError:
        print("plotext is required for test mode. Please install it using 'pip install plotext'.")
        sys.exit(1)

    print("\n-- TEST MODE: Real-Time RMS Line Graph (plotext) --")
    print("Recording 5-second samples in a loop. Press Ctrl+C to exit.\n")
    rms_history = []
    iterations = []
    max_points = 20
    i = 0

    while True:
        audio = capture_audio(RTSP_URL, duration=5)
        if audio is None or len(audio) == 0:
            print("No audio captured, retrying...")
            time.sleep(5)
            continue

        filtered = bandpass_filter(audio, LOWCUT, HIGHCUT, SAMPLING_RATE, FILTER_ORDER)
        rms = measure_rms(filtered)

        rms_history.append(rms)
        iterations.append(i)
        i += 1

        if len(rms_history) > max_points:
            rms_history = rms_history[-max_points:]
            iterations = iterations[-max_points:]

        if rms > NOISE_THRESHOLD_HIGH:
            status = "🔴 ABOVE"
        elif rms < NOISE_THRESHOLD_LOW:
            status = "🔵 BELOW"
        else:
            status = "🟢 OK"

        plt.clf()
        plt.plot(iterations, rms_history, marker="dot", color="cyan")
        plt.horizontal_line(NOISE_THRESHOLD_HIGH, color="red")
        plt.horizontal_line(NOISE_THRESHOLD_LOW, color="blue")
        plt.title("Real-Time RMS (Line Graph)")
        plt.xlabel("Iteration")
        plt.ylabel("RMS")
        plt.ylim(0, max(0.001, max(rms_history) * 1.2))
        plt.show()

        print(f"Current RMS: {rms:.6f} — {status}")
        time.sleep(0.5)

# ---------------------- Dynamic Gain Control Loop ----------------------

def dynamic_gain_control():
    debug_print("Starting dynamic gain controller...", "info")
    set_gain_db(MICROPHONE_NAME, (MIN_GAIN_DB + MAX_GAIN_DB) // 2)

    no_signal_count = 0

    while True:
        audio = capture_audio(RTSP_URL)
        if audio is None or len(audio) == 0:
            debug_print("No audio captured; retrying...", "warning")
            time.sleep(SLEEP_SECONDS)
            continue

        filtered = bandpass_filter(audio, LOWCUT, HIGHCUT, SAMPLING_RATE, FILTER_ORDER)
        rms = measure_rms(filtered)
        debug_print(f"Measured RMS: {rms:.6f}", "info")

        # No-signal detection
        if rms < NO_SIGNAL_THRESHOLD:
            no_signal_count += 1
            debug_print(f"No signal detected ({no_signal_count}/{NO_SIGNAL_COUNT_THRESHOLD})", "warning")
            if no_signal_count >= NO_SIGNAL_COUNT_THRESHOLD:
                debug_print("No signal for too long, executing action...", "error")
                subprocess.call(NO_SIGNAL_ACTION, shell=True)
        else:
            no_signal_count = 0

        current_gain = get_gain_db(MICROPHONE_NAME)
        if current_gain is None:
            debug_print("Failed to read current gain; skipping cycle.", "warning")
            time.sleep(SLEEP_SECONDS)
            continue

        if rms > NOISE_THRESHOLD_HIGH:
            set_gain_db(MICROPHONE_NAME, current_gain - GAIN_STEP_DB)
        elif rms < NOISE_THRESHOLD_LOW:
            set_gain_db(MICROPHONE_NAME, current_gain + GAIN_STEP_DB)

        time.sleep(SLEEP_SECONDS)

# ---------------------- Main ----------------------

def main():
    args = parse_args()

    if args.calibrate:
        mic_params = interactive_calibration()
        proposal = calibrate_and_propose(mic_params)
        save = input("Save these values permanently into the script? [y/N]: ").strip().lower()
        if save in ["y", "yes"]:
            persist_calibration_to_script(os.path.abspath(__file__), proposal)
            print("👍 Calibration values saved. Exiting now.\n")
        else:
            print("❌ Not saving values. Exiting.\n")
        sys.exit(0)

    if args.test:
        test_mode()
        sys.exit(0)

    dynamic_gain_control()

if __name__ == "__main__":
    main()
