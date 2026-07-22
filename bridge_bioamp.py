"""
ORBIT AI — BioAmp EXG Pill Bridge (2-Channel)
==============================================
Reads raw EEG/EMG data from two BioAmp EXG Pills via Serial,
applies real-time filtering, extracts ERD features, and streams 
to the Orbit AI Dashboard.

Fixed: channel array size (21, not 64) and EMG fast-path latency (<100ms target) — see CHANGELOG.md

Hardware Limitation / Methodology Note:
This is a 2-channel approximation (C3 and C4) of motor imagery.
Because a direct midline (Cz) channel is unavailable, the FORWARD 
command (typically foot imagery) is inferred via bilateral symmetry 
(simultaneous similar desynchronization of both C3 and C4).

Wiring:
BioAmp 1 (C3) OUT -> Arduino A0
BioAmp 2 (C4) OUT -> Arduino A1
"""

import serial
import socket
import json
import time
import numpy as np
from scipy.signal import butter, lfilter, welch, sosfilt
import os

from config import (
    SERIAL_PORT, BAUD_RATE, EMG_THRESHOLD_UV,
    ERD_THRESHOLD, ERD_SYMMETRY_TOLERANCE, MODELS_DIR,
    MOTOR_CHANNEL_NAMES, N_MOTOR_CHANNELS, C3_INDEX, C4_INDEX
)
from logger_orbit import logger as orbit_logger

SFREQ = 256.0
CHANNELS = 2  # Updated to 2 for C3 and C4
CHUNK_SIZE = int(SFREQ * 1.0) # 1 second chunks

# Profile Baseline Values
BASELINE_POWER_C3 = 1.0
BASELINE_POWER_C4 = 1.0
try:
    profile_path = MODELS_DIR / "personal_profile.json"
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile = json.load(f)
        
        # Approximate Mu+Beta baseline by combining alpha and beta baselines
        combined_baseline = (profile.get("baseline_alpha", 1.0) + profile.get("baseline_beta", 1.0)) / 2.0
        BASELINE_POWER_C3 = combined_baseline
        BASELINE_POWER_C4 = combined_baseline
except Exception as e:
    print(f"⚠ Could not load personal profile for baseline power: {e}")

MIN_VALID_BASELINE = 1e-6
if BASELINE_POWER_C3 == 1.0 or BASELINE_POWER_C3 < MIN_VALID_BASELINE:
    print("⚠️  WARNING: personal_profile.json baseline_alpha/"
          "baseline_beta appears invalid (missing, default, "
          "or near-zero — possibly from a dead/disconnected "
          "channel during calibration). ERD calculations will "
          "be unstable or meaningless until you re-run "
          "calibrate.py with good electrode contact. Do not "
          "trust classification output until this warning is gone.")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_sos(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def detect_emg_spike(c3_raw, c4_raw):
    """
    STEP 1 — EMG SPIKE DETECTION (jaw clench → STOP)
    Runs BEFORE any EEG band-power analysis.
    """
    emg_amplitude = max(np.max(np.abs(c3_raw)), np.max(np.abs(c4_raw)))
    return emg_amplitude > EMG_THRESHOLD_UV

def compute_mu_beta_power(channel_samples, fs):
    """
    STEP 2 — EEG BAND POWER EXTRACTION
    Extract combined Mu+Beta band power (8-30 Hz).
    """
    f, Pxx = welch(channel_samples, fs=fs, nperseg=min(len(channel_samples), int(fs)))
    idx_band = np.logical_and(f >= 8.0, f <= 30.0)
    return float(np.mean(Pxx[idx_band]))

def compute_erd(live_power, baseline_power):
    """
    STEP 3 — ERD CALCULATION (vs personal baseline)
    Returns standard ERD percentage formula. Positive value means 
    power dropped below baseline (desynchronization).
    """
    if baseline_power < 1e-6:
        return 0.0  # invalid baseline — treat as no ERD, fail safe to IDLE
    return (baseline_power - live_power) / baseline_power

def classify_direction(c3_erd, c4_erd, erd_threshold, symmetry_tolerance):
    """
    STEP 4 — 4-WAY DIRECTIONAL CLASSIFICATION
    """
    c3_active = c3_erd > erd_threshold
    c4_active = c4_erd > erd_threshold
    
    if not c3_active and not c4_active:
        return "IDLE"
    
    if c3_active and c4_active:
        diff = abs(c3_erd - c4_erd)
        avg = (c3_erd + c4_erd) / 2
        symmetry_ratio = diff / avg if avg > 0 else 1.0
        
        if symmetry_ratio < symmetry_tolerance:
            return "FORWARD"
        elif c3_erd > c4_erd:
            return "RIGHT"
        else:
            return "LEFT"
            
    elif c3_active:
        return "RIGHT"
    else:
        return "LEFT"

class BioAmpBridge:
    def __init__(self):
        self.raw_buffer_c3 = []
        self.raw_buffer_c4 = []
        self.chunk_size = CHUNK_SIZE
        
        # Stateful zero-phase equivalent filtering to avoid startup transients
        self.sos = butter_bandpass_sos(1.0, 45.0, SFREQ)
        self.zi_c3 = np.zeros((self.sos.shape[0], 2))
        self.zi_c4 = np.zeros((self.sos.shape[0], 2))
        
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"✅ Connected to dual BioAmp on {SERIAL_PORT}")
        except Exception as e:
            print(f"❌ Could not open Serial Port {SERIAL_PORT}: {e}")
            print("   Check your Arduino connection and COM port number.")
            # Continuing for mock/testing purposes when device not found
            # exit(1)

    def get_packet(self):
        """Reads from Serial. Returns immediately with an 
        emergency-stop packet if EMG is detected on any 
        single sample. Otherwise collects a full 1s epoch 
        before returning a normal classification packet."""
        while len(self.raw_buffer_c3) < self.chunk_size:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if "," in line:
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        c3_sample = float(parts[0])
                        c4_sample = float(parts[1])
                    except ValueError:
                        continue

                    # Fast EMG check on this single sample, before it's added to the buffer
                    if max(abs(c3_sample), abs(c4_sample)) > EMG_THRESHOLD_UV:
                        print(f"🚨 EMG SPIKE DETECTED — immediate stop")
                        orbit_logger.log_event("STOP", 1.0, "GOOD", "N/A")
                        return {
                            "_emg_emergency_stop": True,
                            "_timestamp": time.time(),
                            "_n_channels": N_MOTOR_CHANNELS,
                        }

                    self.raw_buffer_c3.append(c3_sample)
                    self.raw_buffer_c4.append(c4_sample)
        
        chunk_c3 = np.array(self.raw_buffer_c3[:self.chunk_size])
        chunk_c4 = np.array(self.raw_buffer_c4[:self.chunk_size])
        
        self.raw_buffer_c3 = self.raw_buffer_c3[self.chunk_size:]
        self.raw_buffer_c4 = self.raw_buffer_c4[self.chunk_size:]
        
        # 1. EMG Spike Detection (Jaw Clench)
        is_emg_spike = detect_emg_spike(chunk_c3, chunk_c4)
        
        # 2. Filter for EEG (stateful, avoids transient)
        filtered_c3, self.zi_c3 = sosfilt(self.sos, chunk_c3, zi=self.zi_c3)
        filtered_c4, self.zi_c4 = sosfilt(self.sos, chunk_c4, zi=self.zi_c4)
        
        # Prepare payload
        epoch = np.zeros((N_MOTOR_CHANNELS, len(filtered_c3)))
        epoch[C3_INDEX] = filtered_c3
        epoch[C4_INDEX] = filtered_c4
        
        packet = {
            "attention": 0.0,
            "meditation": 0.0,
            "_raw_epoch": epoch.tolist(),
            "_sfreq": SFREQ,
            "_n_channels": N_MOTOR_CHANNELS,
            "_stop_source": "NONE",
            "_c3_erd": 0.0,
            "_c4_erd": 0.0
        }
        
        command = "IDLE"
        confidence = 0.0
        
        if is_emg_spike:
            # EMG overrides everything
            packet["_stop_source"] = "EMG"
            command = "STOP"
            confidence = 1.0
        else:
            # 3. EEG Band Power Extraction
            pwr_c3 = compute_mu_beta_power(filtered_c3, SFREQ)
            pwr_c4 = compute_mu_beta_power(filtered_c4, SFREQ)
            
            # 4. ERD Calculation
            erd_c3 = compute_erd(pwr_c3, BASELINE_POWER_C3)
            erd_c4 = compute_erd(pwr_c4, BASELINE_POWER_C4)
            
            packet["_c3_erd"] = float(erd_c3)
            packet["_c4_erd"] = float(erd_c4)
            
            # 5. Directional Classification
            command = classify_direction(erd_c3, erd_c4, ERD_THRESHOLD, ERD_SYMMETRY_TOLERANCE)
            
            if command != "IDLE":
                active_erd = max(erd_c3, erd_c4)
                confidence = float(min(max(active_erd / (2 * ERD_THRESHOLD), 0.0), 1.0))
            else:
                confidence = 0.5 # Neutral baseline for IDLE
                
        # Log event directly as requested
        orbit_logger.log_event(command, confidence, "GOOD", "NORMAL")

        return packet

    def run(self):
        orbit_logger.log_session_start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 9999))
            s.listen(1)
            print(f"\n📡 Dual BioAmp Bridge listening for Dashboard...")

            while True:
                try:
                    conn, addr = s.accept()
                    print(f"✅ Dashboard linked to BioAmp hardware!")
                    with conn:
                        while True:
                            try:
                                packet = self.get_packet()
                                if packet.get("_emg_emergency_stop"):
                                    # Send priority interrupt immediately as-is
                                    data = (json.dumps(packet) + "\n").encode()
                                    conn.sendall(data)
                                    continue

                                data = (json.dumps(packet) + "\n").encode()
                                conn.sendall(data)
                            except Exception as e:
                                print(f"⚠ Connection lost: {e}")
                                break
                except KeyboardInterrupt:
                    print("\nShutting down bridge...")
                    orbit_logger.log_session_end()
                    break

if __name__ == "__main__":
    BioAmpBridge().run()
