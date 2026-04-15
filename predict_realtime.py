"""
ORBIT AI — BCI Dashboard (MOABB Edition)
==========================================
Connects to EEG Stream Simulator, classifies brainwave states using:
  - Primary:  MOABB CSP+LDA (cross-dataset generalizable) if trained model exists
  - Fallback: BioSensor fixed-threshold (Beta/Alpha ratio)

Run training first:  python train_moabb.py
Then run dashboard:  python predict_realtime.py
"""

import numpy as np
import argparse
import socket
import json
import time
import os
import pickle
from collections import deque
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from scipy.signal import welch

from config import *

COMMANDS = {0: "IDLE", 1: "FORWARD"}

MOABB_MODEL_PATH = MODELS_DIR / "moabb_csp_lda.pkl"
MOABB_STATS_PATH = DATA_DIR   / "moabb_stats.pkl"


class WheelchairArena:
    """2D virtual arena where 'W' moves on FORWARD command."""
    def __init__(self, width=30, height=12):
        self.width  = width
        self.height = height
        self.x = width  // 2
        self.y = height // 2

    def update(self, cmd_id):
        if cmd_id == 1:
            self.y -= 1
            if self.y < 1:
                self.y = self.height - 2

    def render(self):
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.width):
            grid[0][i] = "─"
            grid[self.height - 1][i] = "─"
        for i in range(self.height):
            grid[i][0] = "│"
            grid[i][self.width - 1] = "│"
        grid[0][0] = grid[0][self.width-1] = "+"
        grid[self.height-1][0] = grid[self.height-1][self.width-1] = "+"
        grid[self.y][self.x] = "W"
        return "\n".join("".join(row) for row in grid)


class RealtimePredictor:
    def __init__(self):
        self.console = Console()
        self.arena   = WheelchairArena()
        self.sock_buffer = ""

        # Data Logging
        self.log_file = DATA_DIR / "live_recordings.csv"
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,"
                        "lowGamma,highGamma,attention,meditation,blink,label\n")

        # State
        self.window       = deque(maxlen=TRAIN_WINDOW_SIZE)
        self.last_command = "IDLE"
        self.last_conf    = 0.0
        self.total_fwd    = 0
        self.total_idle   = 0
        self.engine_name  = "⚡ BioSensor"

        # ── Load MOABB Model (if trained) ────────────────────────────
        self.moabb_pipeline = None
        self.moabb_stats    = None

        if MOABB_MODEL_PATH.exists() and MOABB_STATS_PATH.exists():
            try:
                with open(MOABB_MODEL_PATH, 'rb') as f:
                    self.moabb_pipeline = pickle.load(f)
                with open(MOABB_STATS_PATH, 'rb') as f:
                    self.moabb_stats = pickle.load(f)
                acc = self.moabb_stats.get('cv_mean', 0) * 100
                self.engine_name = f"🧠 CSP+LDA ({acc:.0f}%)"
                self.console.print(f"[green]✅ MOABB model loaded — {acc:.1f}% CV accuracy[/green]")
            except Exception as e:
                self.console.print(f"[yellow]⚠ Could not load MOABB model: {e}[/yellow]")
                self.console.print("[yellow]   Falling back to BioSensor engine.[/yellow]")
        else:
            self.console.print("[yellow]⚡ No MOABB model found. Using BioSensor engine.[/yellow]")
            self.console.print("[yellow]   Run 'python train_moabb.py' to train.[/yellow]")

        # ── Load Personal Profile ────────────────────────────────────
        self.profile = None
        self.profile_path = MODELS_DIR / "personal_profile.json"
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f:
                    self.profile = json.load(f)
                self.console.print("[green]✅ Personal Calibration loaded.[/green]")
            except Exception:
                self.console.print("[yellow]⚠ Could not load personal calibration.[/yellow]")
        else:
            self.console.print("[yellow]⚠ No personal profile found. Run calibrate.py for best results.[/yellow]")

        # ── Health & Smoothing State ────────────────────────────────
        self.recent_preds = deque(maxlen=3)
        self.fatigue_history = deque(maxlen=50) # 5 seconds
        self.fatigue_state = "NORMAL"
        self.session_start = time.time()
        self.signal_msg = "AWAITING SIGNAL"

        # ── Connect to Simulator ─────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.console.print("[cyan]📡 Connecting to Simulator...[/cyan]")
        try:
            self.sock.connect(('127.0.0.1', 9999))
            self.console.print("[green]✅ Connected.[/green]")
        except Exception:
            self.console.print("[red]❌ Simulator not running. Start simulate_tgam.py first.[/red]")
            raise SystemExit(1)

    # ── Data Ingestion ───────────────────────────────────────────────
    def get_data(self):
        """Non-blocking read, safely accumulating large JSON packets."""
        try:
            self.sock.setblocking(False)
            while True:
                try:
                    chunk = self.sock.recv(1024 * 512).decode('utf-8', errors='ignore')
                    if not chunk: break
                    self.sock_buffer += chunk
                except BlockingIOError:
                    break
            
            # Find the last complete packet
            if '\n' in self.sock_buffer:
                # Split at the last newline. 
                # [0] contains all complete packets. [1] contains any trailing incomplete chunk.
                parts = self.sock_buffer.rsplit('\n', 1)
                complete_packets = parts[0]
                self.sock_buffer = parts[1]
                
                # We only care about the absolute latest packet for real-time
                last_packet = complete_packets.split('\n')[-1]
                if last_packet:
                    return json.loads(last_packet)
            
            return None
        except Exception:
            return None

    def build_features(self, raw_dict):
        """Extract band power features and log to CSV."""
        keys = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta",
                "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"]
        vals = [float(raw_dict.get(k, 0)) for k in keys]
        with open(self.log_file, "a") as f:
            f.write(",".join(map(str, vals)) + f",{self.last_command}\n")
        vals.extend([0.0] * 7)  # Pad to 18
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
            ch_data = np.array(raw)[0] # C3 channel
            fs = raw_dict.get("_sfreq", 160)
            f, pxx = welch(ch_data, fs=fs, nperseg=min(len(ch_data), int(fs)))
            theta = np.mean(pxx[(f>=4) & (f<=8)])
            alpha = np.mean(pxx[(f>=8) & (f<=13)])
            beta  = np.mean(pxx[(f>=13) & (f<=30)])
            return theta / (alpha + beta + 1e-6)
        except Exception:
            return 0.0

    # ── Classifiers ──────────────────────────────────────────────────
    def classify_moabb(self, raw_dict):
        """
        CSP+LDA classifier using raw multi-channel epoch.
        CSP learns spatial filters → hardware-independent features.
        """
        raw_epoch  = raw_dict.get("_raw_epoch")
        sfreq      = raw_dict.get("_sfreq", 160)
        n_channels = raw_dict.get("_n_channels", 64)

        if raw_epoch is None:
            return self.classify_biosensor(raw_dict)

        epoch = np.array(raw_epoch, dtype=np.float64)  # (n_ch, n_times)

        # Reshape to (1, n_ch, n_times) — what CSP expects
        X = epoch[np.newaxis, :, :]

        try:
            proba = self.moabb_pipeline.predict_proba(X)
            # classes: 0=imagery_left (IDLE), 1=imagery_right (FORWARD)
            forward_prob = float(proba[0][1])
        except Exception:
            # LinearDiscriminantAnalysis with shrinkage doesn't support predict_proba natively.
            # Instead, we use decision_function (distance to boundary) and pass it through a sigmoid.
            try:
                decision = float(self.moabb_pipeline.decision_function(X)[0])
                forward_prob = 1.0 / (1.0 + np.exp(-decision))
            except Exception:
                pred = self.moabb_pipeline.predict(X)[0]
                forward_prob = 1.0 if pred == 1 else 0.0

        return forward_prob, 1.0 - forward_prob

    def classify_biosensor(self, raw_dict):
        """
        Physics-based fixed-threshold fallback.
        Beta/Alpha ratio derived from diagnostic measurements.
        
        eegmat (21ch): REST=-2.97, MATH=-1.78 → threshold = -2.3 works
        NOTE: does NOT generalise across different channel counts.
        """
        x_raw = np.array([list(self.window)], dtype=np.float32)
        attn  = np.mean(x_raw[0, :, 8])
        dist  = attn - (-2.3)
        fwd   = 1.0 / (1.0 + np.exp(-dist * 3.0))
        return fwd, 1.0 - fwd

    # ── UI ───────────────────────────────────────────────────────────
    def build_layout(self):
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))
        layout["body"].split_row(Layout(name="sidebar", ratio=1), Layout(name="arena", ratio=2))

        layout["header"].update(Panel(
            f"[bold cyan]ORBIT AI — BCI WHEELCHAIR INTERFACE[/bold cyan]  [{self.engine_name}]",
            border_style="blue"
        ))

        c = "green" if self.last_command == "FORWARD" else "white"
        if self.fatigue_state == "CRITICAL": c = "red"
        
        info = Table.grid(padding=(0, 1))
        info.add_row("[bold]STATUS:[/bold]",     f"[bold {c}]{self.last_command}[/bold {c}]")
        info.add_row("[bold]CONFIDENCE:[/bold]", f"[yellow]{self.last_conf * 100:.1f}%[/yellow]")
        info.add_row("[bold]SIGNAL:[/bold]",     f"{self.signal_msg}")
        info.add_row("[bold]FATIGUE:[/bold]",    f"[{ 'red' if self.fatigue_state=='CRITICAL' else 'yellow' if self.fatigue_state=='WARNING' else 'green'}]{self.fatigue_state}[/]")
        info.add_row("[bold]FWD/IDLE:[/bold]",   f"[cyan]{self.total_fwd}/{self.total_idle}[/cyan]")
        info.add_row("[bold]ENGINE:[/bold]",     f"[magenta]{self.engine_name}[/magenta]")

        layout["sidebar"].update(Panel(info, title="System Info", border_style="dim"))
        
        arena_color = "red" if self.fatigue_state == "CRITICAL" else "yellow" if self.session_age < 120 else "green"
        arena_title = "Virtual Arena" if self.session_age >= 120 else f"Warmup Phase ({int(self.session_age)}/120s)"
        layout["arena"].update(Panel(self.arena.render(), title=arena_title, border_style=arena_color))
        return layout

    # ── Main Loop ────────────────────────────────────────────────────
    def run(self):
        # Keep last raw_dict for MOABB (needs full packet, not just features)
        self._last_raw = {}

        with Live(self.build_layout(), refresh_per_second=10,
                  screen=True, console=self.console) as live:
            while True:
                raw = self.get_data()
                if raw is None:
                    time.sleep(0.01)
                    live.update(self.build_layout())
                    continue

                self._last_raw = raw
                features = self.build_features(raw)
                self.window.append(features)
                self.session_age = time.time() - self.session_start

                if len(self.window) == TRAIN_WINDOW_SIZE:
                    
                    # 1. Signal Quality Gate
                    valid, self.signal_msg = self.is_signal_valid(raw)
                    if not valid:
                        self.last_command = "IDLE (NO SIGNAL)"
                        live.update(self.build_layout())
                        continue
                        
                    # 2. Fatigue Monitor
                    fatigue_ratio = self.compute_fatigue(raw)
                    self.fatigue_history.append(fatigue_ratio)
                    avg_fatigue = np.mean(self.fatigue_history)
                    
                    fatigue_thresh = self.profile.get("fatigue_threshold", 0.8) if self.profile else 0.8
                    if avg_fatigue > fatigue_thresh * 1.5:
                        self.fatigue_state = "CRITICAL"
                    elif avg_fatigue > fatigue_thresh * 1.2:
                        self.fatigue_state = "WARNING"
                    elif avg_fatigue > fatigue_thresh * 1.05:
                        self.fatigue_state = "ALERT"
                    else:
                        self.fatigue_state = "NORMAL"
                        
                    if self.fatigue_state == "CRITICAL":
                        self.last_command = "STAY_IDLE (FATIGUED)"
                        self.last_conf = 100.0
                        live.update(self.build_layout())
                        continue

                    # 3. Warmup Phase Override
                    if self.session_age < 120:
                        self.last_command = "CALIBRATING"
                        self.signal_msg = "[yellow]WARMUP[/yellow]"
                        live.update(self.build_layout())
                        continue
                        
                    # 4. Engine Classification
                    if self.moabb_pipeline is not None:
                        forward_prob, idle_prob = self.classify_moabb(raw)
                    else:
                        forward_prob, idle_prob = self.classify_biosensor(raw)
                        
                    self.recent_preds.append((forward_prob, idle_prob))

                    # 5. Adaptive Command Smoothing (Weighted Voting)
                    if len(self.recent_preds) == 3:
                        weights = [0.2, 0.3, 0.5]
                        score_fwd = sum([p[0] * w for p, w in zip(self.recent_preds, weights)])
                        
                        threshold = self.profile.get("moabb_confidence_threshold", 0.70) if self.profile else 0.70

                        if score_fwd > threshold:
                            self.last_command = "FORWARD"
                            self.arena.update(1)
                            self.last_conf = score_fwd
                            self.total_fwd += 1
                        else:
                            self.last_command = "IDLE"
                            self.last_conf = 1.0 - score_fwd
                            self.total_idle += 1

                live.update(self.build_layout())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBIT AI — BCI Dashboard")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    RealtimePredictor().run()
