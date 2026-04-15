import socket
import json
import time
import numpy as np
import pickle
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from scipy.signal import welch

console = Console()

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
PROFILE_PATH = MODELS_DIR / "personal_profile.json"

class Calibrator:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_buffer = ""
        self.sfreq = 160.0

    def connect(self):
        console.print("[cyan]📡 Connecting to headset / simulator...[/cyan]")
        try:
            self.sock.connect(('127.0.0.1', 9999))
            self.sock.setblocking(False)
            console.print("[green]✅ Connected successfully![/green]\n")
        except Exception:
            console.print("[red]❌ Connection failed. Ensure simulate_tgam.py or bridge_bioamp.py is running.[/red]")
            raise SystemExit(1)

    def get_data(self):
        """Non-blocking read, safely accumulating large JSON packets."""
        try:
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
                
                # Process the last packet
                last_packet = complete_packets.split('\n')[-1]
                if last_packet:
                    return json.loads(last_packet)
        except Exception:
            return None
        return None

    def compute_bandpower(self, data, sfreq, band):
        """Compute average power in a specific frequency band using Welch's method."""
        # Using channel 0 (which is C3 or the raw bioamp channel)
        ch_data = np.array(data)[0] 
        f, Pxx = welch(ch_data, fs=sfreq, nperseg=min(len(ch_data), int(sfreq)))
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        return np.mean(Pxx[idx_band])

    def record_stage(self, stage_name, duration_sec=10):
        console.print(f"\n[bold yellow]▶ Stage: {stage_name}[/bold yellow]")
        if "REST" in stage_name.upper():
            console.print("Instruction: Relax your body, keep your eyes open, and look straight ahead. Clear your mind.")
        elif "FORWARD" in stage_name.upper():
            console.print("Instruction: Imagine pushing a heavy door forward with your right hand. Focus intensely.")
        elif "CLOSED" in stage_name.upper():
            console.print("Instruction: Close your eyes and take deep breaths. Fully relax.")
        
        console.print("Starting in 3 seconds...")
        time.sleep(3)
        
        start_time = time.time()
        packets = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"[green]Recording {stage_name}...", total=duration_sec)
            
            while time.time() - start_time < duration_sec:
                packet = self.get_data()
                if packet and "_raw_epoch" in packet:
                    packets.append(packet)
                    self.sfreq = packet.get("_sfreq", 160.0)
                
                # Update progress bar
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)
                time.sleep(0.05)
                
        return packets

    def run(self):
        console.print("=" * 60)
        console.print("🧠 ORBIT AI — Personal Calibration System")
        console.print("=" * 60)
        console.print("Creating your distinct brain fingerprint to maximize accuracy.\n")
        
        self.connect()
        
        # We drain the buffer so we have fresh data
        time.sleep(1)
        self.get_data()
        
        # Stages
        rest_packets = self.record_stage("BASELINE (Rest)", 15)
        fwd_packets  = self.record_stage("FORWARD (Motor Imagery)", 15)
        eyes_closed  = self.record_stage("DEEP REST (Eyes Closed)", 15)
        
        console.print("\n[cyan]⚙️ Analyzing brainwaves and calculating thresholds...[/cyan]")
        
        # Analyze Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz)
        bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        
        def avg_band(packets, band_key):
            powers = []
            for p in packets:
                try:
                    pwr = self.compute_bandpower(p["_raw_epoch"], self.sfreq, bands[band_key])
                    if not np.isnan(pwr): powers.append(pwr)
                except Exception:
                    pass
            return float(np.mean(powers)) if powers else 1.0

        # Calculate baselines
        rest_alpha = avg_band(rest_packets, 'alpha')
        rest_beta  = avg_band(rest_packets, 'beta')
        rest_theta = avg_band(rest_packets, 'theta')
        
        fwd_beta = avg_band(fwd_packets, 'beta')
        
        # Profile generation
        # BioSensor dynamic thresholds
        profile = {
            "baseline_alpha": rest_alpha,
            "baseline_beta": rest_beta,
            "baseline_theta": rest_theta,
            "fatigue_threshold": (rest_theta / (rest_alpha + rest_beta)) * 1.5, # 50% above baseline is fatigue
            "forward_beta_multiplier": (fwd_beta / rest_beta) if rest_beta != 0 else 1.2,
            "moabb_confidence_threshold": 0.75, # Starting adaptive threshold
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Ensure minimum bounds
        if profile["forward_beta_multiplier"] < 1.1: profile["forward_beta_multiplier"] = 1.1
        
        with open(PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=4)
            
        console.print(f"\n[green]✅ Calibration Complete![/green]")
        console.print(f"Profile saved to: {PROFILE_PATH}")
        console.print(f"  • Fatigue Ratio Threshold: {profile['fatigue_threshold']:.3f}")
        console.print(f"  • Beta Reactivity: +{((profile['forward_beta_multiplier']-1)*100):.1f}%")
        console.print("\nYou can now run predict_realtime.py. It will auto-load your personal profile.\n")

if __name__ == "__main__":
    Calibrator().run()
