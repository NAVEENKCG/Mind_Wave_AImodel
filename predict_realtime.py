import serial
import time
import torch
import numpy as np
import pandas as pd
import argparse
import socket
import json
from collections import deque
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import BarColumn, Progress

from config import *
from model import EEGClassifier_CNN_LSTM

class WheelchairArena:
    """Displays a 2D map in the terminal where the wheelchair moves."""
    def __init__(self, width=30, height=12):
        self.width = width
        self.height = height
        self.x = width // 2
        self.y = height // 2
        
    def update(self, cmd_id):
        if cmd_id == 1: # FORWARD
            self.y = max(1, self.y - 1)
        elif cmd_id == 2: # LEFT
            self.x = max(1, self.x - 1)
        elif cmd_id == 3: # RIGHT
            self.x = min(self.width - 2, self.x + 1)
        elif cmd_id == 4: # STOP
            pass

    def render(self):
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        # Borders
        for i in range(self.width): grid[0][i] = grid[self.height-1][i] = "─"
        for i in range(self.height): grid[i][0] = grid[i][self.width-1] = "│"
        grid[0][0]=grid[0][self.width-1]=grid[self.height-1][0]=grid[self.height-1][self.width-1]="+"
        
        # Position the Wheelchair
        grid[self.y][self.x] = "[bold yellow]W[/bold yellow]"
        
        res = "\n".join(["".join(row) for row in grid])
        return res

class RealtimePredictor:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arena = WheelchairArena()
        
        # 1. Load Model
        self.model = EEGClassifier_CNN_LSTM(input_size=18, n_classes=5).to(self.device)
        self.model.eval() # Setting to evaluation mode early
        
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.console.print(f"[green]✅ AI Model Loaded Successfully from {MODEL_PATH}[/green]")
        else:
            self.console.print("[yellow]⚠️ No trained model found. Using random weights for demo.[/yellow]")
        
        self.window = deque(maxlen=TRAIN_WINDOW_SIZE)
        self.vote_buffer = deque(maxlen=3)
        self.last_command = "IDLE"

        # 2. Connection Logic
        if self.demo_mode:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.console.print("[cyan]📡 Connecting to Research Data Streamer...[/cyan]")
            try:
                self.sock.connect(('127.0.0.1', 9999))
                self.console.print("[green]✅ Connected to Clinical Data Source.[/green]")
            except:
                self.console.print("[red]❌ ERROR: Simulator not found. Run 'python simulate_tgam.py' first.[/red]")
                exit()
        else:
            try:
                self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                self.console.print(f"[green]✅ Connected to TGAM Headset on {SERIAL_PORT}[/green]")
            except:
                self.console.print("[red]❌ ERROR: TGAM not found. Check SERIAL_PORT in config.py.[/red]")
                exit()

    def get_data(self):
        try:
            if self.demo_mode:
                # Flush the buffer to ensure we get the LATEST signal (removes lag)
                self.sock.setblocking(False)
                latest_packet = None
                while True:
                    try:
                        chunk = self.sock.recv(4096).decode()
                        if not chunk: break
                        packets = chunk.strip().split('\n')
                        if packets:
                            latest_packet = packets[-1] # Get most recent
                    except BlockingIOError:
                        break
                
                if latest_packet:
                    return json.loads(latest_packet)
                return None
            else:
                line = self.ser.readline().decode('utf-8').strip()
                if not line: return None
                vals = [float(x) for x in line.split(',')]
                cols = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta", "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"]
                return dict(zip(cols, vals))
        except:
            return None

    def add_features(self, raw_dict):
        """Add engineered features to match training dimensionality (11 -> 18)."""
        # Exactly match the padding used in train.py
        vals = list(raw_dict.values())[:11]
        vals.extend([0] * 7) # Match exactly the training padding
        return vals

    def create_layout(self, status, confidence, arena_view):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1)
        )
        layout["main"].split_row(
            Layout(name="sidebar", size=40),
            Layout(name="arena_view", ratio=2)
        )
        
        layout["header"].update(Panel("[bold cyan]ORBIT AI - BCI CONTROL INTERFACE[/bold cyan]", border_style="blue"))
        
        # Sidebar Status
        info_table = Table.grid(padding=1)
        info_table.add_row("[bold white]CURRENT COMMAND:[/bold white]", f"[bold yellow]{status}[/bold yellow]")
        
        prog = Progress(BarColumn(bar_width=20), "[progress.percentage]{task.percentage:>3.0f}%")
        prog.add_task("Conf", completed=confidence*100)
        
        layout["sidebar"].update(Panel(info_table, title="System Diagnostics"))
        layout["arena_view"].update(Panel(arena_view, title="Virtual Training Arena"))
        
        return layout

    def run(self):
        with Live(self.console.print("System starting..."), refresh_per_second=4) as live:
            while True:
                raw_data = self.get_data()
                if not raw_data: continue
                
                features = self.add_features(raw_data)
                self.window.append(features)
                
                conf = 0.0
                if len(self.window) == TRAIN_WINDOW_SIZE:
                    input_tensor = torch.FloatTensor(np.array([list(self.window)])).to(self.device)
                    # Use no_grad for inference
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        conf, pred = torch.max(probs, dim=1)
                    
                    cmd_id = pred.item()
                    # Zero-Latency Mode for Funding Pitch
                    self.last_command = COMMANDS[cmd_id]
                    self.arena.update(cmd_id)
                
                live.update(self.create_layout(self.last_command, conf if isinstance(conf, float) else conf.item(), self.arena.render()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    
    predictor = RealtimePredictor(demo_mode=args.demo)
    predictor.run()
