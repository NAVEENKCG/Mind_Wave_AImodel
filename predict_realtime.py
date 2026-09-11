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
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from scipy.signal import welch

from config import *
from config import C3_INDEX, C4_INDEX, N_MOTOR_CHANNELS
from logger_orbit import logger as orbit_logger
from adapt_drift import load_drift_profile, apply_drift_to_threshold, apply_drift_to_fatigue, get_warmup_extension

COMMANDS = {0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "FORWARD"}

MOABB_MODEL_PATH = MODELS_DIR / "moabb_csp_lda.pkl"
MOABB_STATS_PATH = DATA_DIR   / "moabb_stats.pkl"


class BCIArena:
    """2D virtual arena where marker moves based on BCI commands."""
    def __init__(self, width=30, height=12):
        self.width  = width
        self.height = height
        self.x = width  // 2
        self.y = height // 2

    def update(self, cmd):
        if cmd == "FORWARD":
            self.y -= 1
            if self.y < 1:
                self.y = self.height - 2
        elif cmd == "LEFT":
            self.x -= 1
            if self.x < 1:
                self.x = self.width - 2
        elif cmd == "RIGHT":
            self.x += 1
            if self.x > self.width - 2:
                self.x = 1
        # IDLE = no movement

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
        grid[self.y][self.x] = "◆"
        return "\n".join("".join(row) for row in grid)


DEMO_THRESHOLD = 0.45
DEMO_WARMUP = 0


class RealtimePredictor:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        self.console = Console()
        self.arena   = BCIArena()
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

        # ── Load Drift Adaptation Profile (Model 2) ──────────────
        # SAFETY: This NEVER modifies moabb_csp_lda.pkl.
        # It only adjusts thresholds via offsets from drift_profile.json.
        self.drift_profile = load_drift_profile()
        if self.drift_profile is not None:
            ver = self.drift_profile.get('adaptation_version', '?')
            n_sess = self.drift_profile.get('sessions_analyzed', '?')
            self.console.print(
                f"[green]✅ Drift Adaptation v{ver} loaded "
                f"({n_sess} sessions analyzed)[/green]"
            )
        else:
            self.drift_profile = None

        # ── Health & Smoothing State ────────────────────────────────
        self.recent_preds = deque(maxlen=3)
        self.fatigue_history = deque(maxlen=50) # 5 seconds
        self.fatigue_state = "NORMAL"
        self.session_start = time.time()
        self.session_age = 0
        self.signal_msg = "AWAITING SIGNAL"

        # ── Demo Mode Banner ──────────────────────────────────────
        if self.demo_mode:
            self.console.print()
            self.console.print("[bold yellow on black]" + "=" * 42 + "[/]")
            self.console.print("[bold yellow on black]  ORBIT AI  --  DEMO MODE ACTIVE           [/]")
            self.console.print("[bold yellow on black]  Warmup: DISABLED                         [/]")
            self.console.print("[bold yellow on black]  Threshold: 0.45 (relaxed)                [/]")
            self.console.print("[bold yellow on black]  Press Ctrl+C to exit                     [/]")
            self.console.print("[bold yellow on black]" + "=" * 42 + "[/]")
            self.console.print()

        # Start session logging
        orbit_logger.log_session_start()

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
        except Exception:
            return 0.0

    # ── Classifiers ──────────────────────────────────────────────────
    def classify_moabb(self, raw_dict):
        """
        Riemannian classifier using raw multi-channel epoch.
        
        CRITICAL ALIGNMENT (v5.0):
        1. Channel count must match training.
        2. Epoch length must match training.
        3. NO z-score normalization (Riemannian models handle scaling internally).
        4. Use predict() → then derive probability for smoothing.
        """
        raw_epoch  = raw_dict.get("_raw_epoch")
        sfreq      = raw_dict.get("_sfreq", 160)
        n_channels = raw_dict.get("_n_channels", 64)

        if raw_epoch is None:
            return self.classify_biosensor(raw_dict)

        epoch = np.array(raw_epoch, dtype=np.float64)  # (n_ch, n_times)

        # ── Channel alignment ────────────────────────────────────────
        expected_ch = self.moabb_stats.get('n_channels', epoch.shape[0])
        if epoch.shape[0] > expected_ch:
            epoch = epoch[:expected_ch, :]
        elif epoch.shape[0] < expected_ch:
            # Cannot run Riemannian classifier on incomplete data (zero-padding
            # creates degenerate covariance matrices) — fall back to biosensor
            return self.classify_biosensor(raw_dict)

        # ── Epoch length alignment ───────────────────────────────────
        expected_times = self.moabb_stats.get('n_times', epoch.shape[1])
        if epoch.shape[1] > expected_times:
            epoch = epoch[:, :expected_times]
        elif epoch.shape[1] < expected_times:
            repeats = int(np.ceil(expected_times / epoch.shape[1]))
            epoch = np.tile(epoch, (1, repeats))[:, :expected_times]

        # ── Note: No Z-score normalization ───────────────────────────────
        # Riemannian pipelines (Covariances → TS) operate directly on
        # covariance matrices in Riemannian space; z-scoring would destroy
        # the signal's spatial structure.

        # Reshape to (1, n_ch, n_times)
        X = epoch[np.newaxis, :, :]

        # ── Inference: 4-class ────────────────────────────────────────
        class_map = self.moabb_stats.get('classes', COMMANDS)
        try:
            if hasattr(self.moabb_pipeline, 'predict_proba'):
                proba = self.moabb_pipeline.predict_proba(X)[0]
                result = {}
                for i, p in enumerate(proba):
                    cmd_name = class_map.get(i, f"CLASS_{i}")
                    result[cmd_name] = float(p)
                for cmd in ["IDLE", "LEFT", "RIGHT", "FORWARD"]:
                    if cmd not in result:
                        result[cmd] = 0.0
            else:
                pred = int(self.moabb_pipeline.predict(X)[0])
                pred_name = class_map.get(pred, "IDLE")
                result = {"IDLE": 0.0, "LEFT": 0.0, "RIGHT": 0.0, "FORWARD": 0.0}
                result[pred_name] = 0.85
                remaining = 0.15 / max(1, len(result) - 1)
                for k in result:
                    if k != pred_name:
                        result[k] = remaining
            return result
        except Exception:
            return self.classify_biosensor(raw_dict)

    def classify_biosensor(self, raw_dict):
        """
        Physics-based fixed-threshold fallback.
        Beta/Alpha ratio derived from diagnostic measurements.
        
        eegmat (21ch): REST=-2.97, MATH=-1.78 → threshold = -2.3 works
        NOTE: does NOT generalise across different channel counts.
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
                attn = np.mean([list(self.window)[i][8] for i in range(len(self.window))])
        else:
            x_raw = np.array([list(self.window)], dtype=np.float32)
            attn  = np.mean(x_raw[0, :, 8])

        dist  = attn - (-2.3)
        fwd   = 1.0 / (1.0 + np.exp(-dist * 3.0))
        # BioSensor fallback only produces IDLE/FORWARD-like signal
        return {"IDLE": 1.0 - fwd, "LEFT": 0.0, "RIGHT": 0.0, "FORWARD": fwd}

    # ── UI ───────────────────────────────────────────────────────────
    def build_layout(self):
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))
        layout["body"].split_row(Layout(name="sidebar", ratio=1), Layout(name="arena", ratio=2))

        layout["header"].update(Panel(
            f"[bold cyan]ORBIT AI — BCI PROSTHETIC ARM INTERFACE[/bold cyan]  [{self.engine_name}]",
            border_style="blue"
        ))

        c = "green" if self.last_command in ("LEFT", "RIGHT", "FORWARD") else "white"
        if self.fatigue_state == "CRITICAL": c = "red"
        
        info = Table.grid(padding=(0, 1))
        info.add_row("[bold]COMMAND:[/bold]",    f"[bold {c}]{self.last_command}[/bold {c}]")
        info.add_row("[bold]CONFIDENCE:[/bold]", f"[yellow]{self.last_conf * 100:.1f}%[/yellow]")
        info.add_row("[bold]SIGNAL:[/bold]",     f"{self.signal_msg}")
        info.add_row("[bold]FATIGUE:[/bold]",    f"[{ 'red' if self.fatigue_state=='CRITICAL' else 'yellow' if self.fatigue_state=='WARNING' else 'green'}]{self.fatigue_state}[/]")
        info.add_row("[bold]FWD/IDLE:[/bold]",   f"[cyan]{self.total_fwd}/{self.total_idle}[/cyan]")
        info.add_row("[bold]ENGINE:[/bold]",     f"[magenta]{self.engine_name}[/magenta]")

        # ── Demo mode: session timer + keyboard shortcuts ─────────
        if self.demo_mode:
            demo_elapsed = int(time.time() - self.session_start)
            dm, ds = divmod(demo_elapsed, 60)
            dh, dm = divmod(dm, 60)
            timer_str = f"{dh:02d}:{dm:02d}:{ds:02d}"
            info.add_row("")
            info.add_row("[bold yellow]DEMO:[/bold yellow]", f"[yellow]Running: {timer_str}[/yellow]")
            info.add_row("")
            info.add_row("[bold]DEMO CONTROLS:[/bold]")
            info.add_row("  [dim]\\[0][/dim]", "Stream IDLE signal")
            info.add_row("  [dim]\\[1][/dim]", "Stream LIFT signal")
            info.add_row("  [dim italic](via simulate_tgam.py)[/dim italic]")

        layout["sidebar"].update(Panel(info, title="System Info", border_style="dim"))
        
        # ── Brainwave Power Bars ─────────────────────────────────────
        try:
            latest = self.window[-1]
            pwr = Table.grid(padding=(0, 1))
            
            def make_bar(val):
                # The simulator returns power in decibels (dB), which are negative 
                # (usually around -100 to -140 for EEG in Volts). 
                # We shift it by +150 to make it a positive percentage.
                shifted = max(0, val + 150) 
                perc = min(1.0, shifted / 60.0)
                bar_w = 10
                filled = int(bar_w * perc)
                return f"[{'█'*filled}{'░'*(bar_w-filled)}]"

            pwr.add_row("THETA", make_bar(latest[1]))
            pwr.add_row("ALPHA", make_bar((latest[2]+latest[3])/2))
            pwr.add_row("BETA ", make_bar((latest[4]+latest[5])/2))
            
            layout["sidebar"].update(Panel(
                Group(info, "\n[bold]BRAIN POWER[/bold]\n", pwr), 
                title="System Info", border_style="dim"
            ))
        except:
            layout["sidebar"].update(Panel(info, title="System Info", border_style="dim"))
        
        warmup_gate = DEMO_WARMUP if self.demo_mode else 5
        arena_color = "red" if self.fatigue_state == "CRITICAL" else "yellow" if self.session_age < warmup_gate else "green"
        if self.demo_mode:
            arena_title = "Virtual Arena [DEMO]"
        else:
            arena_title = "Virtual Arena" if self.session_age >= warmup_gate else f"Warmup Phase ({int(self.session_age)}/{warmup_gate}s)"
        layout["arena"].update(Panel(self.arena.render(), title=arena_title, border_style=arena_color))
        return layout

    # ── Main Loop ────────────────────────────────────────────────────
    def run(self):
        # Keep last raw_dict for MOABB (needs full packet, not just features)
        self._last_raw = {}

        # Hysteresis state machine: prevents rapid flickering between LIFT/IDLE
        raw_command_streak = {"command": "IDLE", "count": 0}
        confirmed_command = "IDLE"

        try:
          with Live(self.build_layout(), refresh_per_second=10,
                    screen=True, console=self.console) as live:
            while True:
                raw = self.get_data()
                if raw is None:
                    time.sleep(0.05)
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
                    # Model 2: apply drift scaling to fatigue threshold
                    fatigue_thresh = apply_drift_to_fatigue(fatigue_thresh, self.drift_profile)
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
                        self.last_conf = 1.0
                        live.update(self.build_layout())
                        continue

                    warmup_gate = DEMO_WARMUP if self.demo_mode else 5
                    # Model 2: extend warmup if drift engine detected early-session instability
                    warmup_gate += get_warmup_extension(self.drift_profile)
                    if self.session_age < warmup_gate:
                        self.last_command = "CALIBRATING"
                        self.signal_msg = "[yellow]WARMUP[/yellow]"
                        live.update(self.build_layout())
                        continue
                        
                    # 4. Engine Classification (4-class)
                    if self.moabb_pipeline is not None:
                        probs = self.classify_moabb(raw)
                    else:
                        probs = self.classify_biosensor(raw)
                        
                    self.recent_preds.append(probs)

                    # 5. Adaptive Command Smoothing (Weighted Voting — 4-class)
                    if len(self.recent_preds) == 3:
                        weights = [0.2, 0.3, 0.5]
                        smoothed = {"IDLE": 0.0, "LEFT": 0.0, "RIGHT": 0.0, "FORWARD": 0.0}
                        for pred_probs, w in zip(self.recent_preds, weights):
                            for cmd_name in smoothed:
                                smoothed[cmd_name] += pred_probs.get(cmd_name, 0.0) * w
                        
                        # Find best command
                        best_cmd = max(smoothed, key=smoothed.get)
                        best_conf = smoothed[best_cmd]

                        # Apply per-class confidence threshold
                        cmd_to_idx = {"IDLE": 0, "LEFT": 1, "RIGHT": 2, "FORWARD": 3}
                        if self.demo_mode:
                            threshold = DEMO_THRESHOLD
                        else:
                            threshold = CONFIDENCE_THRESHOLDS.get(cmd_to_idx.get(best_cmd, 0), 0.60)
                            if self.profile:
                                threshold = self.profile.get("moabb_confidence_threshold", threshold)
                            # Model 2: apply drift offset to threshold (NEVER changes the model)
                            threshold = apply_drift_to_threshold(threshold, self.drift_profile)

                        # Raw decision: require confidence above threshold
                        if best_cmd == "IDLE" or best_conf < threshold:
                            raw_cmd = "IDLE"
                        else:
                            raw_cmd = best_cmd

                        # Hysteresis state machine: prevents flickering
                        if raw_cmd == raw_command_streak["command"]:
                            raw_command_streak["count"] += 1
                        else:
                            raw_command_streak["command"] = raw_cmd
                            raw_command_streak["count"] = 1

                        CONFIRM_THRESHOLD = 3  # Need 3 consistent non-IDLE to engage
                        RELEASE_THRESHOLD = 1  # Single IDLE immediately stops (safety)

                        if confirmed_command == "IDLE" and raw_cmd != "IDLE":
                            if raw_command_streak["count"] >= CONFIRM_THRESHOLD:
                                confirmed_command = raw_cmd
                        elif confirmed_command != "IDLE" and raw_cmd == "IDLE":
                            if raw_command_streak["count"] >= RELEASE_THRESHOLD:
                                confirmed_command = "IDLE"
                        elif confirmed_command != "IDLE" and raw_cmd != "IDLE" and raw_cmd != confirmed_command:
                            if raw_command_streak["count"] >= CONFIRM_THRESHOLD:
                                confirmed_command = raw_cmd

                        self.last_command = confirmed_command

                        if self.last_command != "IDLE":
                            self.arena.update(self.last_command)
                            self.last_conf = smoothed.get(self.last_command, 0.0)
                            self.total_fwd += 1
                        else:
                            self.last_conf = smoothed.get("IDLE", 0.0)
                            self.total_idle += 1

                        forward_prob = smoothed.get("FORWARD", 0.0)

                        # Log event for session reports (with drift-relevant extras)
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
                            self.last_command, self.last_conf,
                            self.signal_msg, self.fatigue_state,
                            raw_forward_prob=forward_prob,
                            signal_rms=_log_rms,
                            theta_power=_log_theta,
                            alpha_power=_log_alpha,
                            beta_power=_log_beta,
                        )

                live.update(self.build_layout())
        except KeyboardInterrupt:
            pass
        finally:
            orbit_logger.log_session_end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBIT AI — BCI Dashboard")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    RealtimePredictor(demo_mode=args.demo).run()
