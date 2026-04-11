import torch
import numpy as np
import argparse
import socket
import json
import time
import os
from collections import deque
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table

from config import *

# Commands
COMMANDS = {0: "IDLE", 1: "FORWARD"}


class WheelchairArena:
    """2D arena where wheelchair wraps around when it hits the walls."""
    def __init__(self, width=30, height=12):
        self.width = width
        self.height = height
        self.x = width // 2
        self.y = height // 2

    def update(self, cmd_id):
        if cmd_id == 1:  # FORWARD — move up
            self.y -= 1
            if self.y < 1:
                self.y = self.height - 2  # Wrap to bottom

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
    """
    BCI Dashboard — connects to the EEG Stream Simulator,
    classifies brainwave concentration using the Beta/Alpha ratio,
    and drives a virtual wheelchair in real-time.
    """
    def __init__(self):
        self.console = Console()
        self.arena = WheelchairArena()
        
        # Data Logging
        self.log_file = DATA_DIR / "live_recordings.csv"
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,"
                        "lowGamma,highGamma,attention,meditation,blink,label\n")

        self.window = deque(maxlen=TRAIN_WINDOW_SIZE)
        self.last_command = "IDLE"
        self.last_conf = 0.0
        self.total_fwd = 0
        self.total_idle = 0

        # Connect to Simulator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.console.print("[cyan]📡 Connecting to Simulator...[/cyan]")
        try:
            self.sock.connect(('127.0.0.1', 9999))
            self.console.print("[green]✅ Connected.[/green]")
        except:
            self.console.print("[red]❌ Simulator not running. Start simulate_tgam.py first.[/red]")
            raise SystemExit(1)

    def get_data(self):
        """Read the latest packet from the socket (non-blocking)."""
        try:
            self.sock.setblocking(False)
            latest = None
            while True:
                try:
                    chunk = self.sock.recv(4096).decode()
                    if not chunk: break
                    packets = [p for p in chunk.strip().split('\n') if p]
                    if packets: latest = packets[-1]
                except BlockingIOError: break
            return json.loads(latest) if latest else None
        except: return None

    def build_features(self, raw_dict):
        """Extract the 11 EEG features and log to CSV."""
        keys = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta",
                "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"]
        vals = [float(raw_dict.get(k, 0)) for k in keys]
        
        # Log to file for future model fine-tuning
        with open(self.log_file, "a") as f:
            f.write(",".join(map(str, vals)) + f",{self.last_command}\n")
            
        vals.extend([0.0] * 7)  # Pad to 18 features
        return vals

    def classify(self, x_raw):
        """
        Physics-based BioSensor classifier.
        
        Uses the Beta/Alpha ratio (attention feature at index 8) — a
        domain-independent biomarker of concentration. This is the same
        principle used by commercial BCI headsets (NeuroSky, Emotiv).
        
        Fixed threshold derived from diagnostic measurements:
          REST: attention mean = -2.97 dB  →  sigmoid → ~12% (IDLE)
          MATH: attention mean = -1.78 dB  →  sigmoid → ~83% (FORWARD)
        """
        # Attention = mean(Beta_dB) - mean(Alpha_dB) for each timestep
        attn_values = x_raw[0, :, 8]  # shape: (TRAIN_WINDOW_SIZE,)
        mean_attn = np.mean(attn_values)
        
        # Fixed threshold — no calibration, no adaptation
        THRESHOLD = -2.3
        
        # Sigmoid mapping: distance > 0 = concentration, < 0 = rest
        distance = mean_attn - THRESHOLD
        forward_prob = 1.0 / (1.0 + np.exp(-distance * 3.0))
        
        return forward_prob, 1.0 - forward_prob

    def build_layout(self):
        """Build the Rich terminal UI."""
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))
        layout["body"].split_row(Layout(name="sidebar", ratio=1), Layout(name="arena", ratio=2))
        
        layout["header"].update(Panel(
            "[bold cyan]ORBIT AI — BCI WHEELCHAIR INTERFACE[/bold cyan]  [⚡ BioSensor Engine]",
            border_style="blue"
        ))
        
        cmd_color = "green" if self.last_command == "FORWARD" else "white"
        info = Table.grid(padding=(0, 1))
        info.add_row("[bold]STATUS:[/bold]", f"[bold {cmd_color}]{self.last_command}[/bold {cmd_color}]")
        info.add_row("[bold]CONFIDENCE:[/bold]", f"[yellow]{self.last_conf * 100:.1f}%[/yellow]")
        info.add_row("[bold]SAMPLES:[/bold]", f"[green]{os.path.getsize(self.log_file)//1024} KB[/green]")
        info.add_row("[bold]FWD/IDLE:[/bold]", f"[cyan]{self.total_fwd}/{self.total_idle}[/cyan]")
        
        layout["sidebar"].update(Panel(info, title="System Info", border_style="dim"))
        layout["arena"].update(Panel(self.arena.render(), title="Virtual Arena", border_style="green"))
        return layout

    def run(self):
        """Main prediction loop."""
        with Live(self.build_layout(), refresh_per_second=10, screen=True, console=self.console) as live:
            while True:
                raw = self.get_data()
                if raw is None:
                    time.sleep(0.01)
                    live.update(self.build_layout())
                    continue

                features = self.build_features(raw)
                self.window.append(features)

                if len(self.window) == TRAIN_WINDOW_SIZE:
                    x_raw = np.array([list(self.window)], dtype=np.float32)
                    forward_prob, idle_prob = self.classify(x_raw)

                    if forward_prob > 0.50:
                        self.last_command = "FORWARD"
                        self.arena.update(1)
                        self.last_conf = (self.last_conf * 0.3) + (forward_prob * 0.7)
                        self.total_fwd += 1
                    else:
                        self.last_command = "IDLE"
                        self.last_conf = (self.last_conf * 0.3) + (idle_prob * 0.7)
                        self.total_idle += 1

                live.update(self.build_layout())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBIT AI — BCI Dashboard")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()
    RealtimePredictor().run()
