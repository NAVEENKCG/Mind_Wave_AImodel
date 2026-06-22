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
        self.engine_name  = "⚡ BioSensor"

        # Broadcast state for Websockets
        self.broadcast_data = {
            "command": "IDLE",
            "confidence": 0.0,
            "signal": "AWAITING SIGNAL",
            "fatigue": "NORMAL",
            "power": {"theta": 0.0, "alpha": 0.0, "beta": 0.0}
        }

        # Load Models (Simplified)
        self.moabb_pipeline = None
        self.moabb_stats    = None
        if MOABB_MODEL_PATH.exists() and MOABB_STATS_PATH.exists():
            try:
                with open(MOABB_MODEL_PATH, 'rb') as f: self.moabb_pipeline = pickle.load(f)
                with open(MOABB_STATS_PATH, 'rb') as f: self.moabb_stats = pickle.load(f)
                acc = self.moabb_stats.get('cv_mean', 0) * 100
                self.engine_name = f"🧠 CSP+LDA ({acc:.0f}%)"
            except: pass

        self.profile = None
        self.profile_path = MODELS_DIR / "personal_profile.json"
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f: self.profile = json.load(f)
            except: pass

        self.recent_preds = deque(maxlen=3)
        self.fatigue_history = deque(maxlen=50)
        self.fatigue_state = "NORMAL"
        self.session_start = time.time()
        self.session_age = 0
        self.signal_msg = "AWAITING SIGNAL"

        orbit_logger.log_session_start()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("📡 Connecting to Simulator on 9999...")
        try:
            self.sock.connect(('127.0.0.1', 9999))
            print("✅ Connected to simulator.")
        except Exception:
            print("❌ Simulator not running. Start simulate_tgam.py first.")
            raise SystemExit(1)

        # Start WebSocket server
        self.connected_clients = set()
        self.ws_thread = threading.Thread(target=self.start_ws_server, daemon=True)
        self.ws_thread.start()

    def start_ws_server(self):
        async def handler(websocket):
            print("🌐 Web Dashboard Connected!")
            self.connected_clients.add(websocket)
            try:
                while True:
                    await websocket.send(json.dumps(self.broadcast_data))
                    await asyncio.sleep(0.1)
            except websockets.exceptions.ConnectionClosed:
                print("🌐 Web Dashboard Disconnected!")
            finally:
                self.connected_clients.remove(websocket)

        async def main():
            print("🚀 WebSocket Server running on ws://localhost:8765")
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
        keys = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta", "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"]
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

    def classify_biosensor(self, raw_dict):
        x_raw = np.array([list(self.window)], dtype=np.float32)
        attn  = np.mean(x_raw[0, :, 8])
        dist  = attn - (-2.3)
        fwd   = 1.0 / (1.0 + np.exp(-dist * 3.0))
        return fwd, 1.0 - fwd

    def run(self):
        try:
            while True:
                raw = self.get_data()
                if raw is None:
                    time.sleep(0.05)
                    continue

                features = self.build_features(raw)
                self.window.append(features)
                self.session_age = time.time() - self.session_start

                # Update Brain Power data
                latest = self.window[-1]
                theta_pwr = max(0, latest[1] + 150) / 60.0
                alpha_pwr = max(0, ((latest[2]+latest[3])/2) + 150) / 60.0
                beta_pwr = max(0, ((latest[4]+latest[5])/2) + 150) / 60.0

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

                    if self.session_age < 5:
                        self.last_command = "CALIBRATING"
                        self.broadcast_data.update({"command": "IDLE", "signal": "WARMUP"})
                        continue
                        
                    forward_prob, idle_prob = self.classify_biosensor(raw)
                    self.recent_preds.append((forward_prob, idle_prob))

                    if len(self.recent_preds) == 3:
                        weights = [0.2, 0.3, 0.5]
                        score_fwd = sum([p[0] * w for p, w in zip(self.recent_preds, weights)])
                        threshold = self.profile.get("moabb_confidence_threshold", 0.50) if self.profile else 0.50

                        if score_fwd > threshold:
                            self.last_command = "FORWARD"
                            self.last_conf = score_fwd
                        else:
                            self.last_command = "IDLE"
                            self.last_conf = 1.0 - score_fwd

                        self.broadcast_data.update({
                            "command": self.last_command,
                            "confidence": float(self.last_conf),
                            "signal": self.signal_msg,
                            "fatigue": self.fatigue_state
                        })

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            orbit_logger.log_session_end()

if __name__ == "__main__":
    RealtimePredictor().run()
