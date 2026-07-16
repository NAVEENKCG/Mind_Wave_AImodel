"""
ORBIT AI — BCI WebSocket Server
==========================================
Connects to EEG Stream Simulator and broadcasts brainwave states
over WebSockets for the Next.js Virtual Wheelchair Dashboard.
"""

import numpy as np
import argparse
import socket
import json
import time
import pickle
import threading
import asyncio
import websockets
from collections import deque
from pathlib import Path
from scipy.signal import welch

from config import *
from logger_orbit import logger as orbit_logger

COMMANDS = {0: "IDLE", 1: "FORWARD"}

MOABB_MODEL_PATH = MODELS_DIR / "moabb_csp_lda.pkl"
MOABB_STATS_PATH = DATA_DIR   / "moabb_stats.pkl"

class RealtimePredictor:
    def __init__(self):
        self.sock_buffer = ""
        self.log_file = DATA_DIR / "live_recordings.csv"

        # State
        self.window       = deque(maxlen=TRAIN_WINDOW_SIZE)
        self.last_command = "IDLE"
        self.last_conf    = 0.0
        self.total_fwd    = 0
        self.total_idle   = 0
        self.engine_name  = "BioSensor"

        # Broadcast state for Websockets
        self.broadcast_data = {
            "command": "IDLE",
            "confidence": 0.0,
            "signal": "AWAITING SIGNAL",
            "fatigue": "NORMAL",
            "power": {"theta": 0.0, "alpha": 0.0, "beta": 0.0}
        }

        # Load Models
        self.moabb_pipeline = None
        self.moabb_stats    = None
        if MOABB_MODEL_PATH.exists() and MOABB_STATS_PATH.exists():
            try:
                with open(MOABB_MODEL_PATH, 'rb') as f: self.moabb_pipeline = pickle.load(f)
                with open(MOABB_STATS_PATH, 'rb') as f: self.moabb_stats = pickle.load(f)
                acc = self.moabb_stats.get('cv_mean', 0) * 100
                pipe_name = self.moabb_stats.get('pipeline', 'CSP+LDA')
                is_calibrated = self.moabb_stats.get('calibrated', False)
                uses_fbcsp = self.moabb_stats.get('uses_fbcsp', False)
                cal_tag = " [Platt-calibrated]" if is_calibrated else ""
                fbcsp_tag = " [FBCSP]" if uses_fbcsp else ""
                self.engine_name = f"{pipe_name}{cal_tag} ({acc:.0f}%)"
                print(f"[MODEL] Loaded MOABB {pipe_name} pipeline{cal_tag}{fbcsp_tag} (CV accuracy: {acc:.1f}%)")
                print(f"[MODEL] Expected input: {self.moabb_stats.get('n_channels')} channels x {self.moabb_stats.get('n_times')} samples")
            except Exception as e:
                print(f"[WARN] Failed to load MOABB model: {e}")
                self.moabb_pipeline = None

        self.profile = None
        self.profile_path = MODELS_DIR / "personal_profile.json"
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f: self.profile = json.load(f)
            except: pass

        self.recent_preds = deque(maxlen=3)  # Sharper smoothing: less regression-to-mean
        self.fatigue_history = deque(maxlen=50)
        self.fatigue_state = "NORMAL"
        self.session_start = time.time()
        self.session_age = 0
        self.signal_msg = "AWAITING SIGNAL"

        # Adaptive baseline for BioSensor fallback
        self.baseline_attention_values = []
        self.baseline_computed = False
        self.adaptive_center = -2.3  # Will be updated during warmup

        # Hysteresis state machine: prevents rapid flickering between FORWARD/IDLE
        self.confirmed_command = "IDLE"
        self.consecutive_same = 0
        self.CONFIRM_THRESHOLD = 2    # Need N consistent predictions to switch TO forward
        self.RELEASE_THRESHOLD = 2    # Symmetric: same responsiveness for both transitions

        orbit_logger.log_session_start()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("[CONN] Connecting to Simulator on 9999...")
        try:
            self.sock.connect(('127.0.0.1', 9999))
            print("[CONN] Connected to simulator.")
        except Exception:
            print("[ERROR] Simulator not running. Start simulate_tgam.py first.")
            raise SystemExit(1)

        # Start WebSocket server
        self.connected_clients = set()
        self.ws_thread = threading.Thread(target=self.start_ws_server, daemon=True)
        self.ws_thread.start()

    def start_ws_server(self):
        async def handler(websocket):
            print("[WS] Web Dashboard Connected!")
            self.connected_clients.add(websocket)
            try:
                while True:
                    await websocket.send(json.dumps(self.broadcast_data))
                    await asyncio.sleep(0.1)
            except websockets.exceptions.ConnectionClosed:
                print("[WS] Web Dashboard Disconnected!")
            finally:
                self.connected_clients.remove(websocket)

        async def main():
            print("[WS] WebSocket Server running on ws://localhost:8765")
            async with websockets.serve(handler, "localhost", 8765):
                await asyncio.Future()

        asyncio.run(main())

    def get_data(self):
        try:
            self.sock.setblocking(False)
            while True:
                try:
                    chunk = self.sock.recv(1024 * 512).decode('utf-8', errors='ignore')
                    if not chunk: break
                    self.sock_buffer += chunk
                except BlockingIOError:
                    break
            
            if '\n' in self.sock_buffer:
                parts = self.sock_buffer.rsplit('\n', 1)
                complete_packets = parts[0]
                self.sock_buffer = parts[1]
                last_packet = complete_packets.split('\n')[-1]
                if last_packet: return json.loads(last_packet)
            return None
        except: return None

    def build_features(self, raw_dict):
        keys = ["delta", "theta", "alpha", "beta", "gamma", "attention", "meditation", "blink"]
        vals = [float(raw_dict.get(k, 0)) for k in keys]
        return vals

    def is_signal_valid(self, raw_dict):
        raw = raw_dict.get("_raw_epoch")
        if not raw: return False, "NO DATA"
        flat = np.array(raw).flatten()
        if np.all(flat == 0.0): return False, "NO CONTACT"
        if np.std(flat) > 300: return False, "NOISY"
        return True, "GOOD"

    def compute_fatigue(self, raw_dict):
        raw = raw_dict.get("_raw_epoch")
        if not raw: return 0.0
        try:
            ch_data = np.array(raw)[0]
            fs = raw_dict.get("_sfreq", 160)
            f, pxx = welch(ch_data, fs=fs, nperseg=min(len(ch_data), int(fs)))
            theta = np.mean(pxx[(f>=4) & (f<=8)])
            alpha = np.mean(pxx[(f>=8) & (f<=13)])
            beta  = np.mean(pxx[(f>=13) & (f<=30)])
            return theta / (alpha + beta + 1e-6)
        except: return 0.0

    # ── Classification: MOABB Riemannian Pipeline (Primary) ──────────
    def classify_moabb(self, raw_dict):
        """
        Use the trained MOABB Riemannian pipeline on raw multi-channel epoch data.
        Returns (forward_prob, idle_prob) or None on failure.
        """
        raw_epoch = raw_dict.get("_raw_epoch")
        if raw_epoch is None:
            return None

        X = np.array(raw_epoch)
        expected_ch = self.moabb_stats.get('n_channels', 64)
        expected_t  = self.moabb_stats.get('n_times', 160)

        if X.shape[0] != expected_ch:
            return None

        if X.shape[1] != expected_t:
            if X.shape[1] > expected_t:
                X = X[:, :expected_t]
            else:
                return None

        X = X[np.newaxis, :, :]

        try:
            if hasattr(self.moabb_pipeline, 'predict_proba'):
                proba = self.moabb_pipeline.predict_proba(X)[0]
                idle_prob = proba[0]
                fwd_prob  = proba[1] if len(proba) > 1 else 1.0 - idle_prob
            else:
                pred = self.moabb_pipeline.predict(X)[0]
                fwd_prob  = 0.85 if pred == 1 else 0.15
                idle_prob = 1.0 - fwd_prob

            return fwd_prob, idle_prob
        except Exception as e:
            print(f"[WARN] MOABB classification failed: {e}")
            return None

    # ── Classification: BioSensor Adaptive (Fallback) ────────────────
    def classify_biosensor(self, raw_dict):
        """
        Fallback classifier using motor-cortex-focused band power attention
        with adaptive thresholding.
        """
        raw_epoch = raw_dict.get("_raw_epoch")

        if raw_epoch is not None:
            epoch_arr = np.array(raw_epoch)
            fs = raw_dict.get("_sfreq", 160)
            n_ch = epoch_arr.shape[0]

            if n_ch >= 64:
                motor_indices = [7, 9, 11, 5, 13]
            elif n_ch >= 32:
                motor_indices = [3, 5, 7]
            else:
                motor_indices = list(range(n_ch))

            attn_values = []
            for ch_idx in motor_indices:
                if ch_idx >= n_ch:
                    continue
                ch_data = epoch_arr[ch_idx]
                nperseg = min(len(ch_data), int(fs))
                if nperseg < 8:
                    continue
                f, psd = welch(ch_data, fs=fs, nperseg=nperseg)
                alpha = np.mean(psd[(f >= 8) & (f <= 13)])
                beta  = np.mean(psd[(f >= 13) & (f <= 30)])
                ratio = np.log10((beta + 1e-12) / (alpha + 1e-12))
                attn_values.append(ratio)

            if attn_values:
                attn = np.mean(attn_values)
            else:
                attn = np.mean([list(self.window)[i][5] for i in range(len(self.window))])
        else:
            x_raw = np.array([list(self.window)], dtype=np.float32)
            attn  = np.mean(x_raw[0, :, 5])

        dist = attn - self.adaptive_center
        fwd  = 1.0 / (1.0 + np.exp(-dist * 5.0))
        return fwd, 1.0 - fwd

    # ── Hysteresis: Stabilize command output ─────────────────────────
    def apply_hysteresis(self, raw_command):
        """
        Prevents rapid flickering between FORWARD and IDLE.
        Requires multiple consecutive consistent predictions before switching state.
        """
        if raw_command == self.confirmed_command:
            # Same as current state — reinforce
            self.consecutive_same += 1
        else:
            # Different from current state — count toward switch
            self.consecutive_same += 1
            
            # Determine how many consecutive predictions are needed to switch
            if self.confirmed_command == "IDLE" and raw_command == "FORWARD":
                needed = self.CONFIRM_THRESHOLD  # Switch TO forward: need N confirmations
            else:
                needed = self.RELEASE_THRESHOLD  # Switch BACK to idle: need more confirmations
            
            if self.consecutive_same >= needed:
                self.confirmed_command = raw_command
                self.consecutive_same = 0
            else:
                # Not enough evidence to switch yet — keep current state
                return self.confirmed_command
        
        return self.confirmed_command

    def run(self):
        # Track consecutive raw predictions for hysteresis
        raw_command_streak = {"command": "IDLE", "count": 0}
        
        try:
            while True:
                raw = self.get_data()
                if raw is None:
                    time.sleep(0.05)
                    continue

                features = self.build_features(raw)
                self.window.append(features)
                self.session_age = time.time() - self.session_start

                # Update Brain Power data (5-band: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma)
                latest = self.window[-1]
                theta_pwr = max(0, latest[1] + 150) / 60.0
                alpha_pwr = max(0, latest[2] + 150) / 60.0
                beta_pwr  = max(0, latest[3] + 150) / 60.0

                self.broadcast_data["power"] = {
                    "theta": min(1.0, theta_pwr),
                    "alpha": min(1.0, alpha_pwr),
                    "beta": min(1.0, beta_pwr)
                }

                if len(self.window) == TRAIN_WINDOW_SIZE:
                    valid, self.signal_msg = self.is_signal_valid(raw)
                    if not valid:
                        self.last_command = "IDLE"
                        self.broadcast_data.update({"command": "IDLE", "signal": self.signal_msg})
                        continue
                        
                    fatigue_ratio = self.compute_fatigue(raw)
                    self.fatigue_history.append(fatigue_ratio)
                    avg_fatigue = np.mean(self.fatigue_history)
                    
                    fatigue_thresh = self.profile.get("fatigue_threshold", 0.8) if self.profile else 0.8
                    if avg_fatigue > fatigue_thresh * 1.5: self.fatigue_state = "CRITICAL"
                    elif avg_fatigue > fatigue_thresh * 1.2: self.fatigue_state = "WARNING"
                    elif avg_fatigue > fatigue_thresh * 1.05: self.fatigue_state = "ALERT"
                    else: self.fatigue_state = "NORMAL"
                        
                    if self.fatigue_state == "CRITICAL":
                        self.last_command = "IDLE"
                        self.broadcast_data.update({"command": "IDLE", "fatigue": self.fatigue_state, "signal": "FATIGUED"})
                        continue

                    # During warmup: collect baseline attention values for adaptive threshold
                    if self.session_age < 3:
                        self.last_command = "CALIBRATING"
                        self.broadcast_data.update({"command": "IDLE", "signal": "WARMUP"})
                        attn_val = features[5]  # attention index in 5-band layout
                        self.baseline_attention_values.append(attn_val)
                        continue

                    # Compute adaptive baseline once after warmup completes
                    if not self.baseline_computed and len(self.baseline_attention_values) > 3:
                        self.adaptive_center = float(np.median(self.baseline_attention_values))
                        self.baseline_computed = True
                        print(f"[ADAPT] Baseline attention center: {self.adaptive_center:.3f}")

                    # ── Classification: MOABB first, BioSensor fallback ──
                    result = None
                    if self.moabb_pipeline is not None:
                        result = self.classify_moabb(raw)
                        
                    if result is not None:
                        forward_prob, idle_prob = result
                    else:
                        forward_prob, idle_prob = self.classify_biosensor(raw)

                    self.recent_preds.append((forward_prob, idle_prob))

                    if len(self.recent_preds) == self.recent_preds.maxlen:
                        # Sharper exponential weights: latest prediction dominates
                        weights = [1, 2, 4]  # ~57% latest, ~29% middle, ~14% oldest
                        total_w = sum(weights)
                        score_fwd = sum(p[0] * w for p, w in zip(self.recent_preds, weights)) / total_w
                        threshold = self.profile.get("moabb_confidence_threshold", 0.50) if self.profile else 0.50

                        # Raw decision from the model
                        raw_cmd = "FORWARD" if score_fwd > threshold else "IDLE"
                        
                        # Apply hysteresis to stabilize the command
                        if raw_cmd == raw_command_streak["command"]:
                            raw_command_streak["count"] += 1
                        else:
                            raw_command_streak["command"] = raw_cmd
                            raw_command_streak["count"] = 1

                        # Require consecutive consistent predictions before switching
                        if self.confirmed_command == "IDLE" and raw_cmd == "FORWARD":
                            if raw_command_streak["count"] >= self.CONFIRM_THRESHOLD:
                                self.confirmed_command = "FORWARD"
                        elif self.confirmed_command == "FORWARD" and raw_cmd == "IDLE":
                            if raw_command_streak["count"] >= self.RELEASE_THRESHOLD:
                                self.confirmed_command = "IDLE"

                        self.last_command = self.confirmed_command
                        self.last_conf = score_fwd if self.last_command == "FORWARD" else (1.0 - score_fwd)

                        self.broadcast_data.update({
                            "command": self.last_command,
                            "confidence": float(self.last_conf),
                            "signal": self.signal_msg,
                            "fatigue": self.fatigue_state
                        })

                        # Log every classification event to the session logger
                        orbit_logger.log_event(
                            command=self.last_command,
                            confidence=float(self.last_conf),
                            signal_quality=self.signal_msg,
                            fatigue_state=self.fatigue_state,
                        )

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            orbit_logger.log_session_end()

if __name__ == "__main__":
    RealtimePredictor().run()
