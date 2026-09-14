"""
ORBIT AI — BCI WebSocket Server (v7.0)
==========================================
Connects to EEG Stream Simulator and broadcasts brainwave states
over WebSockets for the Next.js Virtual Prosthetic Arm Dashboard.

v7.0: Artifact rejection, confidence-weighted EWMA, adaptive threshold.
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
import sys
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from train_moabb import FBCSPTransformer, BandpassFilter
    sys.modules['__main__'].FBCSPTransformer = FBCSPTransformer
    sys.modules['__main__'].BandpassFilter = BandpassFilter
except Exception as e:
    pass

from config import *
from config import C3_INDEX, C4_INDEX, N_MOTOR_CHANNELS
from logger_orbit import logger as orbit_logger
from adapt_drift import load_drift_profile, apply_drift_to_threshold, apply_drift_to_fatigue, get_warmup_extension
from universal_model import get_universal_model, UniversalModel

COMMANDS = {0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "FORWARD"}
# NOTE: Artifact threshold is in µV. Simulator data arrives in Volts (MNE convention),
# so we convert to Volts before comparison. BioAmp bridge data is also scaled to Volts
# inside its bandpass filter output. If raw ADC µV data is ever passed directly,
# this conversion would need to be conditional.
ARTIFACT_THRESHOLD_UV = 100  # ±100µV peak-to-peak artifact rejection

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
            "probabilities": {"IDLE": 1.0, "LEFT": 0.0, "RIGHT": 0.0, "FORWARD": 0.0},
            "signal": "AWAITING SIGNAL",
            "fatigue": "NORMAL",
            "power": {"theta": 0.0, "alpha": 0.0, "beta": 0.0}
        }

        # Load Model 1 (Universal Brain)
        self.universal_model = get_universal_model()
        self.moabb_pipeline = self.universal_model.pipeline
        self.moabb_stats = self.universal_model.stats
        if self.universal_model.is_loaded:
            acc = self.universal_model.cv_accuracy * 100
            pipe_name = self.universal_model.pipeline_name
            is_calibrated = self.moabb_stats.get('calibrated', False)
            uses_fbcsp = self.moabb_stats.get('uses_fbcsp', False)
            cal_tag = " [Platt-calibrated]" if is_calibrated else ""
            fbcsp_tag = " [FBCSP]" if uses_fbcsp else ""
            self.engine_name = f"{pipe_name}{cal_tag} ({acc:.0f}%)"
            print(f"[MODEL] Loaded Model 1 (Universal Brain) {pipe_name} pipeline{cal_tag}{fbcsp_tag} (CV accuracy: {acc:.1f}%)")
            print(f"[MODEL] Expected input: {self.universal_model.expected_channels} channels x {self.universal_model.expected_times} samples")
        else:
            self.engine_name = "BioSensor"

        self.profile = None
        self.profile_path = MODELS_DIR / "personal_profile.json"
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f: self.profile = json.load(f)
            except: pass

        # ── Load Drift Adaptation Profile (Model 2) ──────────────
        # SAFETY: This NEVER modifies moabb_csp_lda.pkl.
        # It only adjusts thresholds via offsets from drift_profile.json.
        self.drift_profile = load_drift_profile()
        if self.drift_profile is not None:
            ver = self.drift_profile.get('adaptation_version', '?')
            n_sess = self.drift_profile.get('sessions_analyzed', '?')
            print(f"[MODEL2] Drift Adaptation v{ver} loaded ({n_sess} sessions analyzed)")
        else:
            self.drift_profile = None

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
        self.warmup_probs = []  # Collect model probabilities during warmup for adaptive threshold
        self.adaptive_threshold = 0.60  # Will be updated after warmup (raised from 0.50 for safety)

        # Hysteresis state machine: prevents rapid flickering between LIFT/IDLE
        self.confirmed_command = "IDLE"
        self.consecutive_same = 0
        self.CONFIRM_THRESHOLD = 3    # Need 3 consistent LIFTs to engage (deliberate intent)
        self.RELEASE_THRESHOLD = 1    # Single IDLE immediately stops (safety: fast release)

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

        def process_request(connection, request):
            from websockets.http11 import Response
            upgrade_header = request.headers.get("Upgrade", "").lower()
            if upgrade_header != "websocket":
                from http import HTTPStatus
                body = json.dumps({
                    "status": "online",
                    "service": "ORBIT AI BCI WebSocket Server",
                    "websocket_url": "ws://localhost:8765",
                    "note": "This port is for WebSocket connections. To view the dashboard, open the Next.js app in your browser (e.g. http://localhost:3000)."
                }).encode()
                return Response(HTTPStatus.OK, "OK", websockets.Headers([("Content-Type", "application/json")]), body)
            return None

        async def main():
            print("[WS] WebSocket Server running on ws://localhost:8765")
            print("[NOTE] To view the Prosthetic Arm Dashboard, open your web app (http://localhost:3000).")
            print("[NOTE] Do not open ws://localhost:8765 directly in your browser URL bar.")
            async with websockets.serve(handler, "localhost", 8765, process_request=process_request):
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
        epoch = np.array(raw)
        flat = epoch.flatten()
        if np.all(flat == 0.0): return False, "NO CONTACT"
        if np.std(flat) > 300: return False, "NOISY"
        # Artifact rejection: check peak-to-peak per channel
        ptp = np.ptp(epoch, axis=1)  # (n_channels,)
        threshold_v = ARTIFACT_THRESHOLD_UV * 1e-6  # convert µV to V
        bad_ratio = np.mean(ptp > threshold_v)
        if bad_ratio > 0.25:
            return False, "ARTIFACT"
        return True, "GOOD"

    def compute_fatigue(self, raw_dict):
        raw = raw_dict.get("_raw_epoch")
        if not raw: return 0.0
        try:
            epoch = np.array(raw)
            fs = raw_dict.get("_sfreq", 160)
            ratios = []
            for ch_idx in [C3_INDEX, C4_INDEX]:
                if ch_idx >= epoch.shape[0]:
                    continue
                ch_data = epoch[ch_idx]
                if np.all(ch_data == 0):
                    continue
                f, pxx = welch(ch_data, fs=fs, nperseg=min(len(ch_data), int(fs)))
                theta = np.mean(pxx[(f>=4) & (f<=8)])
                alpha = np.mean(pxx[(f>=8) & (f<=13)])
                beta  = np.mean(pxx[(f>=13) & (f<=30)])
                ratios.append(theta / (alpha + beta + 1e-6))
            return float(np.mean(ratios)) if ratios else 0.0
        except Exception: return 0.0

    # ── Classification: MOABB Riemannian Pipeline (Primary) ──────────
    def classify_moabb(self, raw_dict):
        """
        Use Model 1 (Universal Brain) Riemannian pipeline on raw multi-channel epoch data.
        Returns a dict {"IDLE": p, "LEFT": p, "RIGHT": p, "FORWARD": p} or None on failure.
        """
        raw_epoch = raw_dict.get("_raw_epoch")
        if raw_epoch is None or not self.universal_model.is_loaded:
            return None

        sfreq = raw_dict.get("_sfreq", 160)
        return self.universal_model.predict_proba(np.array(raw_epoch), sfreq=sfreq)

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

            if n_ch == N_MOTOR_CHANNELS:
                # 21-channel BioAmp bridge: only C3 and C4 have real data
                motor_indices = [C3_INDEX, C4_INDEX]
            elif n_ch >= 64:
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
                # Skip channels that are entirely padded with zeros (e.g. from 2-channel bioamp bridge)
                if np.all(ch_data == 0):
                    continue
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
                    # Model 2: apply drift scaling to fatigue threshold
                    fatigue_thresh = apply_drift_to_fatigue(fatigue_thresh, self.drift_profile)
                    if avg_fatigue > fatigue_thresh * 1.5: self.fatigue_state = "CRITICAL"
                    elif avg_fatigue > fatigue_thresh * 1.2: self.fatigue_state = "WARNING"
                    elif avg_fatigue > fatigue_thresh * 1.05: self.fatigue_state = "ALERT"
                    else: self.fatigue_state = "NORMAL"
                        
                    if self.fatigue_state == "CRITICAL":
                        self.last_command = "IDLE"
                        self.broadcast_data.update({"command": "IDLE", "fatigue": self.fatigue_state, "signal": "FATIGUED"})
                        continue

                    # During warmup: collect baseline attention values for adaptive threshold
                    warmup_duration = 5 + get_warmup_extension(self.drift_profile)
                    if self.session_age < warmup_duration:
                        self.last_command = "CALIBRATING"
                        self.broadcast_data.update({"command": "IDLE", "signal": "WARMUP"})
                        # Compute beta/alpha ratio from raw epoch (not the pre-computed
                        # attention field, which is 0.0 when data comes from bridge_bioamp)
                        raw_epoch = raw.get("_raw_epoch")
                        if raw_epoch is not None:
                            epoch_arr = np.array(raw_epoch)
                            fs = raw.get("_sfreq", 160)
                            warmup_ratios = []
                            for ch_idx in [C3_INDEX, C4_INDEX]:
                                if ch_idx >= epoch_arr.shape[0]:
                                    continue
                                ch_data = epoch_arr[ch_idx]
                                if np.all(ch_data == 0):
                                    continue
                                nperseg = min(len(ch_data), int(fs))
                                if nperseg < 8:
                                    continue
                                f, psd = welch(ch_data, fs=fs, nperseg=nperseg)
                                alpha = np.mean(psd[(f >= 8) & (f <= 13)])
                                beta  = np.mean(psd[(f >= 13) & (f <= 30)])
                                warmup_ratios.append(np.log10((beta + 1e-12) / (alpha + 1e-12)))
                            if warmup_ratios:
                                self.baseline_attention_values.append(float(np.mean(warmup_ratios)))
                        # Also collect model probabilities during warmup for adaptive threshold
                        if self.moabb_pipeline is not None:
                            warmup_result = self.classify_moabb(raw)
                            if warmup_result is not None:
                                self.warmup_probs.append(warmup_result[0])  # forward_prob
                        continue

                    # Compute adaptive baseline once after warmup completes
                    if not self.baseline_computed and len(self.baseline_attention_values) > 3:
                        self.adaptive_center = float(np.median(self.baseline_attention_values))
                        self.baseline_computed = True
                        print(f"[ADAPT] Baseline attention center: {self.adaptive_center:.3f}")
                        # Set adaptive threshold from warmup probability distribution
                        if len(self.warmup_probs) > 3:
                            sorted_probs = sorted(self.warmup_probs)
                            # Threshold at 90th percentile of resting-state forward probabilities
                            # (raised from 75th for prosthetic arm safety — reduces false positives)
                            idx_90 = int(len(sorted_probs) * 0.90)
                            self.adaptive_threshold = max(0.60, min(0.75, sorted_probs[idx_90]))
                            print(f"[ADAPT] Adaptive LIFT threshold: {self.adaptive_threshold:.3f} (from {len(self.warmup_probs)} warmup predictions)")
                        else:
                            self.adaptive_threshold = 0.60
                            print(f"[ADAPT] Using default threshold: {self.adaptive_threshold:.2f}")

                    # ── Classification: MOABB first, BioSensor fallback ──
                    result = None
                    if self.moabb_pipeline is not None:
                        result = self.classify_moabb(raw)
                        
                    if result is not None:
                        probs = result  # {"IDLE": p, "LEFT": p, "RIGHT": p, "FORWARD": p}
                    else:
                        # BioSensor fallback: only produces IDLE/FORWARD-like signal
                        fwd_p, idle_p = self.classify_biosensor(raw)
                        probs = {"IDLE": idle_p, "LEFT": 0.0, "RIGHT": 0.0, "FORWARD": fwd_p}

                    self.recent_preds.append(probs)

                    if len(self.recent_preds) == self.recent_preds.maxlen:
                        # Confidence-weighted EWMA across all 4 classes
                        base_weights = [1, 2, 4]  # recency weights
                        smoothed = {"IDLE": 0.0, "LEFT": 0.0, "RIGHT": 0.0, "FORWARD": 0.0}
                        total_w = 0
                        for pred_probs, bw in zip(self.recent_preds, base_weights):
                            conf = max(pred_probs.values())  # prediction confidence
                            w = bw * conf
                            for cmd_name in smoothed:
                                smoothed[cmd_name] += pred_probs.get(cmd_name, 0.0) * w
                            total_w += w
                        for cmd_name in smoothed:
                            smoothed[cmd_name] /= (total_w + 1e-9)

                        # Find best non-IDLE command
                        best_cmd = max(smoothed, key=smoothed.get)
                        best_conf = smoothed[best_cmd]

                        # Apply per-class confidence threshold
                        cmd_to_idx = {"IDLE": 0, "LEFT": 1, "RIGHT": 2, "FORWARD": 3}
                        threshold = CONFIDENCE_THRESHOLDS.get(cmd_to_idx.get(best_cmd, 0), 0.60)
                        # Model 2: apply drift offset to threshold (NEVER changes the model)
                        threshold = apply_drift_to_threshold(threshold, self.drift_profile)

                        # Raw decision: require confidence above class-specific threshold
                        if best_cmd == "IDLE" or best_conf < threshold:
                            raw_cmd = "IDLE"
                        else:
                            raw_cmd = best_cmd
                        
                        # Apply hysteresis to stabilize the command
                        if raw_cmd == raw_command_streak["command"]:
                            raw_command_streak["count"] += 1
                        else:
                            raw_command_streak["command"] = raw_cmd
                            raw_command_streak["count"] = 1

                        # Require consecutive consistent predictions before switching
                        if self.confirmed_command == "IDLE" and raw_cmd != "IDLE":
                            if raw_command_streak["count"] >= self.CONFIRM_THRESHOLD:
                                self.confirmed_command = raw_cmd
                        elif self.confirmed_command != "IDLE" and raw_cmd == "IDLE":
                            if raw_command_streak["count"] >= self.RELEASE_THRESHOLD:
                                self.confirmed_command = "IDLE"
                        elif self.confirmed_command != "IDLE" and raw_cmd != "IDLE" and raw_cmd != self.confirmed_command:
                            # Switching between non-IDLE commands requires confirmation too
                            if raw_command_streak["count"] >= self.CONFIRM_THRESHOLD:
                                self.confirmed_command = raw_cmd

                        self.last_command = self.confirmed_command
                        self.last_conf = smoothed.get(self.last_command, 0.0)
                        forward_prob = smoothed.get("FORWARD", 0.0)

                        self.broadcast_data.update({
                            "command": self.last_command,
                            "confidence": float(self.last_conf),
                            "probabilities": {k: round(v, 4) for k, v in smoothed.items()},
                            "signal": self.signal_msg,
                            "fatigue": self.fatigue_state
                        })

                        # Log every classification event to the session logger
                        # Compute band powers and signal RMS for drift engine
                        _log_rms = 0.0
                        _log_theta = _log_alpha = _log_beta = 0.0
                        try:
                            _raw_epoch = raw.get("_raw_epoch")
                            if _raw_epoch is not None:
                                _epoch_arr = np.array(_raw_epoch)
                                _log_rms = float(np.sqrt(np.mean(_epoch_arr ** 2)))
                                _fs = raw.get("_sfreq", 160)
                                for _ci in [C3_INDEX, C4_INDEX]:
                                    if _ci < _epoch_arr.shape[0]:
                                        _cd = _epoch_arr[_ci]
                                        if not np.all(_cd == 0):
                                            _f, _pxx = welch(_cd, fs=_fs, nperseg=min(len(_cd), int(_fs)))
                                            _log_theta += float(np.mean(_pxx[(_f >= 4) & (_f <= 8)]))
                                            _log_alpha += float(np.mean(_pxx[(_f >= 8) & (_f <= 13)]))
                                            _log_beta += float(np.mean(_pxx[(_f >= 13) & (_f <= 30)]))
                                _log_theta /= 2.0
                                _log_alpha /= 2.0
                                _log_beta /= 2.0
                        except Exception:
                            pass

                        orbit_logger.log_event(
                            command=self.last_command,
                            confidence=float(self.last_conf),
                            signal_quality=self.signal_msg,
                            fatigue_state=self.fatigue_state,
                            raw_forward_prob=forward_prob,
                            signal_rms=_log_rms,
                            theta_power=_log_theta,
                            alpha_power=_log_alpha,
                            beta_power=_log_beta,
                        )

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            orbit_logger.log_session_end()

if __name__ == "__main__":
    RealtimePredictor().run()
