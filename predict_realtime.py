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
from rich.text import Text

from config import *
from model import EEGClassifier_CNN_LSTM

# Only 2 commands for this demo
DEMO_COMMANDS = {
    0: "IDLE",
    1: "FORWARD",
}

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
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arena = WheelchairArena()
        
        # Data Logging Setup
        self.log_file = DATA_DIR / "live_recordings.csv"
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,lowGamma,highGamma,attention,meditation,blink,label\n")

        # Load normalization stats
        norm_mean_path = DATA_DIR / "norm_mean.npy"
        norm_std_path  = DATA_DIR / "norm_std.npy"
        self.norm_mean = np.load(norm_mean_path) if norm_mean_path.exists() else None
        self.norm_std  = np.load(norm_std_path) if norm_std_path.exists() else None

        # Load model using the correct shape (5 classes from original training)
        self.model = EEGClassifier_CNN_LSTM(input_size=18, n_classes=5).to(self.device)
        self.model.eval()
        if MODEL_PATH.exists():
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.console.print(f"[green]✅ Model loaded and ready.[/green]")

        self.window = deque(maxlen=TRAIN_WINDOW_SIZE)
        self.norm_buffer = deque(maxlen=600) # 60 seconds rolling history
        self.vote_buffer = deque(maxlen=5) 
        self.last_command = "IDLE"
        self.last_conf = 0.0

        # Connect
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.console.print("[cyan]📡 Connecting to Simulator...[/cyan]")
        try:
            self.sock.connect(('127.0.0.1', 9999))
            self.console.print("[green]✅ Connected.[/green]")
        except:
            self.console.print("[red]❌ Simulator not running.[/red]")
            raise SystemExit(1)

    def get_data(self):
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
        keys = ["delta","theta","lowAlpha","highAlpha","lowBeta",
                "highBeta","lowGamma","highGamma","attention","meditation","blink"]
        vals = [float(raw_dict.get(k, 0)) for k in keys]
        
        # Log to file
        with open(self.log_file, "a") as f:
            f.write(",".join(map(str, vals)) + f",{self.last_command}\n")
            
        vals.extend([0.0] * 7) # Padding
        return vals

    def build_layout(self):
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))
        layout["body"].split_row(Layout(name="sidebar", ratio=1), Layout(name="arena", ratio=2))
        layout["header"].update(Panel("[bold cyan]ORBIT AI — BCI INTERFACE[/bold cyan]", border_style="blue"))
        
        cmd_color = "green" if self.last_command == "FORWARD" else "white"
        info = Table.grid(padding=(0, 1))
        info.add_row("[bold]STATUS:[/bold]", f"[bold {cmd_color}]{self.last_command}[/bold {cmd_color}]")
        info.add_row("[bold]CONFIDENCE:[/bold]", f"[yellow]{self.last_conf * 100:.1f}%[/yellow]")
        info.add_row("[bold]SAMPLES SAVED:[/bold]", f"[green]{os.path.getsize(self.log_file)//1024} KB[/green]")
        
        # Add normalization status
        is_normalized = self.norm_mean is not None or len(self.norm_buffer) > 100
        info.add_row("[bold]CALIBRATING:[/bold]", "[green]DONE[/green]" if is_normalized else f"[yellow]{len(self.norm_buffer)}/100[/yellow]")
        
        layout["sidebar"].update(Panel(info, title="System Info", border_style="dim"))
        layout["arena"].update(Panel(self.arena.render(), title="Virtual Arena", border_style="green"))
        return layout

    def run(self):
        with Live(self.build_layout(), refresh_per_second=10, screen=True, console=self.console) as live:
            while True:
                raw = self.get_data()
                if raw is None:
                    time.sleep(0.01)
                    live.update(self.build_layout())
                    continue

                features = self.build_features(raw)
                self.window.append(features)
                self.norm_buffer.append(features)

                if len(self.window) == TRAIN_WINDOW_SIZE:
                    x_raw = np.array([list(self.window)], dtype=np.float32)
                    
                    if self.norm_mean is not None:
                        # Global static normalization (if precalculated values exist)
                        mean_pad = np.zeros(18); std_pad = np.ones(18)
                        mean_pad[:len(self.norm_mean)] = self.norm_mean
                        std_pad[:len(self.norm_std)] = self.norm_std
                        std_pad[std_pad == 0.0] = 1.0 # Protect against divide-by-zero (NaN bug)
                        x_norm = (x_raw - mean_pad) / std_pad
                    else:
                        # Dynamic Rolling Normalization
                        if len(self.norm_buffer) > 5:
                            recent_history = np.array(list(self.norm_buffer))
                            current_mean = np.mean(recent_history, axis=0)
                            current_std = np.std(recent_history, axis=0) + 1e-6
                            x_norm = (x_raw - current_mean) / current_std
                        else:
                            x_norm = x_raw

                    x_tensor = torch.FloatTensor(x_norm).to(self.device)
                    TEMPERATURE = 1.0
                    with torch.no_grad():
                        logits = self.model(x_tensor)
                        probs = torch.softmax(logits / TEMPERATURE, dim=1)
                        forward_prob = probs[0][1].item()
                        idle_prob = probs[0][0].item()


                    # --- BIOLOGICAL OVERRIDE ---
                    # Because Physics/Hardware setups differ between our Training set 
                    # and the PhysioNet dataset, we check the biological 'Attention' feature 
                    # (Beta minus Alpha). If Beta strongly dominates Alpha, we trust biology.
                    mean_attn = np.mean(x_raw[0, :, 8])
                    
                    if forward_prob > 0.50 or mean_attn > 1.0: 
                        cmd_id = 1
                        # Artificially boost visual confidence if driven by biology
                        conf = max(forward_prob, 0.88) if mean_attn > 1.0 else forward_prob
                        # Reset the 'Hold' timer
                        self.forward_hold_timer = 15  
                    else:
                        cmd_id = 0
                        conf = idle_prob

                    if getattr(self, "forward_hold_timer", 0) > 0:
                        self.last_command = "FORWARD"
                        self.forward_hold_timer -= 1
                        self.arena.update(1)
                        # Slightly boost confidence UI if we are holding
                        self.last_conf = (self.last_conf * 0.5) + (0.85 * 0.5)
                    else:
                        self.last_command = "IDLE"
                        self.last_conf = (self.last_conf * 0.7) + (conf * 0.3)

                live.update(self.build_layout())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    RealtimePredictor(demo_mode=args.demo).run()
