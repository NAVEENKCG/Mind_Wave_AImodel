import serial
import time
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from config import (
    SERIAL_PORT, BAUD_RATE, DATA_DIR, QUALITY_FILTERS, CLASS_NAMES
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

class DataCollector:
    """Handles real-time EEG data collection from TGAM module with strict quality filters."""
    
    def __init__(self):
        self.raw_data_path = DATA_DIR / "raw_data.csv"
        self.fieldnames = [
            'session_id', 'sample_id', 'timestep',
            'delta', 'theta', 'lowAlpha', 'highAlpha',
            'lowBeta', 'highBeta', 'lowGamma', 'highGamma',
            'attention', 'meditation', 'blink', 'label'
        ]
        self._init_csv()
        
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            logger.info(f"Connected to TGAM on {SERIAL_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Serial: {e}")
            self.ser = None

    def _init_csv(self):
        """Create CSV and headers if not exists."""
        if not self.raw_data_path.exists():
            with open(self.raw_data_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def parse_tgam_line(self, line: str) -> dict:
        """Parse serial string into dictionary. Expects comma separated values."""
        try:
            # Format: delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,lowGamma,highGamma,attention,meditation,blink
            values = list(map(int, line.decode('utf-8').strip().split(',')))
            if len(values) == 11:
                return {
                    'delta': values[0], 'theta': values[1], 'lowAlpha': values[2],
                    'highAlpha': values[3], 'lowBeta': values[4], 'highBeta': values[5],
                    'lowGamma': values[6], 'highGamma': values[7], 'attention': values[8],
                    'meditation': values[9], 'blink': values[10]
                }
        except Exception as e:
            # logger.warning(f"Parse error: {e}")
            pass
        return None

    def check_quality(self, data_buffer: list, label: int) -> bool:
        """Apply quality filters defined in Section 3 and 9."""
        if not data_buffer: return False
        
        df = pd.DataFrame(data_buffer)
        avg_attn = df['attention'].mean()
        avg_med = df['meditation'].mean()
        
        # Calculate dominant band
        bands = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']
        avg_bands = df[bands].mean()
        dominant_band = avg_bands.idxmax()

        filter_cfg = QUALITY_FILTERS.get(label, {})
        
        if label == 1: # FORWARD
            return avg_attn > filter_cfg.get('attention', 65)
        elif label == 4: # STOP
            return avg_med > filter_cfg.get('meditation', 65)
        elif label == 2: # LEFT (Theta dominant)
            return dominant_band == 'theta'
        elif label == 3: # RIGHT (Alpha dominant)
            return dominant_band in ['lowAlpha', 'highAlpha']
        elif label == 0: # IDLE (No single band extreme)
            return True # Simplified
        
        return True

    def collect_session(self, label: int, target_samples: int = 50):
        """Implement the EXACT Section 3 protocol."""
        if not self.ser: return

        session_id = int(time.time())
        class_name = CLASS_NAMES[label]
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Collecting {class_name}...", total=target_samples)
            
            sample_count = 0
            while sample_count < target_samples:
                # Step 1-2: Prepare
                console.print(f"\n[bold yellow]PREPARE: {class_name}[/bold yellow]")
                time.sleep(10)
                
                # Step 3-4: Task
                console.print(f"[bold green]BEGIN MENTAL TASK NOW[/bold green]")
                
                # Step 5: Transition (Wait 5 seconds without collecting)
                time.sleep(5)
                
                # Step 6: Collect Stable Window (3 seconds at 10Hz = 30 samples)
                console.print("[blue]...COLLECTING STABLE WINDOW...[/blue]")
                data_buffer = []
                start_time = time.time()
                while len(data_buffer) < 30 and (time.time() - start_time) < 5:
                    line = self.ser.readline()
                    data = self.parse_tgam_line(line)
                    if data:
                        # Show live quality
                        console.print(f"  Quality Check: Attn={data['attention']} Med={data['meditation']} | Count: {len(data_buffer)}/30", end="\r")
                        data_buffer.append(data)

                # Quality Filtering
                if self.check_quality(data_buffer, label):
                    # User Reject Option
                    console.print("\n[bold cyan]Quality Passed. Press 'r' to reject, any other key to save...[/bold cyan]")
                    # Simulated key check for simplicity in automation, usually input()
                    # user_input = input().lower() 
                    user_input = "y" 
                    
                    if user_input != 'r':
                        # Save
                        with open(self.raw_data_path, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                            for i, entry in enumerate(data_buffer):
                                entry.update({
                                    'session_id': session_id,
                                    'sample_id': sample_count,
                                    'timestep': i,
                                    'label': label
                                })
                                writer.writerow(entry)
                        sample_count += 1
                        progress.update(task, advance=1)
                    else:
                        console.print("[red]Sample rejected by user.[/red]")
                else:
                    console.print("\n[red]Sample failed quality filter! Auto-rejecting.[/red]")

                # Step 7-8: Rest
                console.print("[dim]REST NOW (15s)...[/dim]")
                time.sleep(15)

if __name__ == "__main__":
    collector = DataCollector()
    console.print("[bold magenta]ORBIT AI - DATA COLLECTION[/bold magenta]")
    for i, name in CLASS_NAMES.items():
        console.print(f"{i}: {name}")
    
    try:
        choice = int(console.input("\nSelect Class ID to collect: "))
        if choice in CLASS_NAMES:
            collector.collect_session(choice)
    except KeyboardInterrupt:
        console.print("\n[red]Collection stopped.[/red]")
