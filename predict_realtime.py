import torch
import numpy as np
import argparse
import socket
import json
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
        if cmd_id == 1:  # FORWARD — move up, wrap around
            self.y -= 1
            if self.y < 1:
                self.y = self.height - 2  # Wrap to bottom

    def render(self):
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        # Borders
        for i in range(self.width):
            grid[0][i] = "─"
            grid[self.height - 1][i] = "─"
        for i in range(self.height):
            grid[i][0] = "│"
            grid[i][self.width - 1] = "│"
        grid[0][0] = grid[0][self.width-1] = "+"
        grid[self.height-1][0] = grid[self.height-1][self.width-1] = "+"
        # Wheelchair marker
        grid[self.y][self.x] = "W"
        return "\n".join("".join(row) for row in grid)


class RealtimePredictor:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arena = WheelchairArena()

        # Load normalization stats if they exist
        norm_mean_path = DATA_DIR / "norm_mean.npy"
        norm_std_path  = DATA_DIR / "norm_std.npy"
        if norm_mean_path.exists() and norm_std_path.exists():
            self.norm_mean = np.load(norm_mean_path)
            self.norm_std  = np.load(norm_std_path)
        else:
            self.norm_mean = None
            self.norm_std  = None

        # Load model
        self.model = EEGClassifier_CNN_LSTM(input_size=18, n_classes=2).to(self.device)
        self.model.eval()
        if MODEL_PATH.exists():
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device)
            )
            self.console.print(f"[green]✅ Model loaded from {MODEL_PATH}[/green]")
        else:
            self.console.print("[yellow]⚠ No trained model found — using random weights.[/yellow]")

        self.window = deque(maxlen=TRAIN_WINDOW_SIZE)
        self.last_command = "IDLE"
        self.last_conf = 0.0

        # Connect
        if self.demo_mode:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.console.print("[cyan]📡 Connecting to Simulator...[/cyan]")
            try:
                self.sock.connect(('127.0.0.1', 9999))
                self.console.print("[green]✅ Connected.[/green]")
            except Exception:
                self.console.print("[red]❌ Simulator not running. Start simulate_tgam.py first.[/red]")
                raise SystemExit(1)

    def get_data(self):
        """Read latest packet from socket, flushing stale data."""
        try:
            if self.demo_mode:
                self.sock.setblocking(False)
                latest = None
                while True:
                    try:
                        chunk = self.sock.recv(4096).decode()
                        if not chunk:
                            break
                        packets = [p for p in chunk.strip().split('\n') if p]
                        if packets:
                            latest = packets[-1]
                    except BlockingIOError:
                        break
                return json.loads(latest) if latest else None
        except Exception:
            return None

    def build_features(self, raw_dict):
        """Convert raw packet → exactly 18 features (zeros for padding)."""
        keys = ["delta","theta","lowAlpha","highAlpha","lowBeta",
                "highBeta","lowGamma","highGamma","attention","meditation","blink"]
        vals = [float(raw_dict.get(k, 0)) for k in keys]
        vals.extend([0.0] * 7)  # Pad to 18
        return vals

    def build_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="arena",   ratio=2)
        )

        # Header
        layout["header"].update(
            Panel("[bold cyan]ORBIT AI — BCI CONTROL INTERFACE[/bold cyan]",
                  border_style="blue")
        )

        # Sidebar
        cmd_color = "green" if self.last_command == "FORWARD" else "white"
        info = Table.grid(padding=(0, 1))
        info.add_row("[bold]STATUS:[/bold]",
                     f"[bold {cmd_color}]{self.last_command}[/bold {cmd_color}]")
        info.add_row("[bold]CONFIDENCE:[/bold]",
                     f"[yellow]{self.last_conf * 100:.1f}%[/yellow]")
        info.add_row("[bold]DEVICE:[/bold]",
                     "[cyan]Clinical EEG (Demo)[/cyan]")
        layout["sidebar"].update(
            Panel(info, title="System Diagnostics", border_style="dim")
        )

        # Arena
        layout["arena"].update(
            Panel(self.arena.render(), title="Virtual Wheelchair Arena",
                  border_style="green")
        )
        return layout

    def run(self):
        with Live(self.build_layout(), refresh_per_second=10,
                  screen=True, console=self.console) as live:
            while True:
                raw = self.get_data()
                if raw is None:
                    live.update(self.build_layout())
                    continue

                features = self.build_features(raw)
                self.window.append(features)

                if len(self.window) == TRAIN_WINDOW_SIZE:
                    x = torch.FloatTensor(
                        np.array([list(self.window)])
                    ).to(self.device)

                    with torch.no_grad():
                        probs = torch.softmax(self.model(x), dim=1)
                        conf, pred = torch.max(probs, dim=1)

                    cmd_id = pred.item()
                    self.last_conf = conf.item()

                    # Map to IDLE or FORWARD only
                    if cmd_id == 1:
                        self.last_command = "FORWARD"
                        self.arena.update(1)
                    else:
                        self.last_command = "IDLE"

                live.update(self.build_layout())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    predictor = RealtimePredictor(demo_mode=args.demo)
    predictor.run()
