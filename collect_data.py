import socket
import json
import time
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from config import DATA_DIR, CLASS_NAMES

console = Console()

class UniversalCollector:
    """Collects data from the Network Bridge (Socket) instead of raw Serial."""
    
    def __init__(self):
        self.raw_data_path = DATA_DIR / "personal_raw_data.csv"
        self.fieldnames = [
            'session_id', 'sample_id', 'timestep',
            'delta', 'theta', 'lowAlpha', 'highAlpha',
            'lowBeta', 'highBeta', 'lowGamma', 'highGamma',
            'attention', 'meditation', 'blink', 'label'
        ]
        self._init_csv()
        self.sock_buffer = ""

    def _init_csv(self):
        if not self.raw_data_path.exists():
            with open(self.raw_data_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect(('127.0.0.1', 9999))
            self.sock.setblocking(False)
            console.print("[green]✅ Connected to ORBIT Bridge![/green]")
        except Exception:
            console.print("[red]❌ Bridge not found. Run bridge_bioamp.py or simulate_tgam.py first.[/red]")
            raise SystemExit(1)

    def get_packet(self):
        try:
            while True:
                try:
                    chunk = self.sock.recv(1024 * 512).decode('utf-8', errors='ignore')
                    if not chunk: break
                    self.sock_buffer += chunk
                except BlockingIOError: break
            
            if '\n' in self.sock_buffer:
                parts = self.sock_buffer.rsplit('\n', 1)
                self.sock_buffer = parts[1]
                return json.loads(parts[0].split('\n')[-1])
        except Exception: return None
        return None

    def collect_session(self, label: int, target_samples: int = 20):
        session_id = int(time.time())
        class_name = CLASS_NAMES[label]
        
        console.print(f"\n[bold yellow]PREPARING FOR: {class_name}[/bold yellow]")
        time.sleep(2)
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Collecting {class_name}...", total=target_samples)
            
            count = 0
            while count < target_samples:
                packet = self.get_packet()
                if packet:
                    # Save the logic
                    with open(self.raw_data_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                        # We map the 11 base values + label
                        row = {
                            'session_id': session_id,
                            'sample_id': count,
                            'timestep': 0,
                            'delta': packet.get('delta',0),
                            'theta': packet.get('theta',0),
                            'lowAlpha': packet.get('lowAlpha',0),
                            'highAlpha': packet.get('highAlpha',0),
                            'lowBeta': packet.get('lowBeta',0),
                            'highBeta': packet.get('highBeta',0),
                            'lowGamma': packet.get('lowGamma',0),
                            'highGamma': packet.get('highGamma',0),
                            'attention': packet.get('attention',0),
                            'meditation': packet.get('meditation',0),
                            'blink': packet.get('blink',0),
                            'label': label
                        }
                        writer.writerow(row)
                    count += 1
                    progress.update(task, advance=1)
                time.sleep(0.1)

if __name__ == "__main__":
    collector = UniversalCollector()
    collector.connect()
    console.print("\n[bold magenta]ORBIT AI - UNIVERSAL DATA COLLECTION[/bold magenta]")
    for i, name in CLASS_NAMES.items():
        console.print(f"{i}: {name}")
    
    try:
        choice = int(console.input("\nSelect Class to record: "))
        collector.collect_session(choice)
        console.print("[green]Session saved to personal_raw_data.csv![/green]")
    except KeyboardInterrupt:
        pass
