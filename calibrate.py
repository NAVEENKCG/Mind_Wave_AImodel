import socket
import json
import time
import numpy as np
import pickle
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from scipy.signal import welch

console = Console()

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
PROFILE_PATH = MODELS_DIR / "personal_profile.json"

MAX_CONNECT_ATTEMPTS = 3
RETRY_DELAY_SEC = 2
SIGNAL_CHECK_PACKETS = 5


class Calibrator:
    def __init__(self):
        self.sock = None
        self.sock_buffer = ""
        self.sfreq = 160.0

    # ── CHANGE 1: Startup message ────────────────────────────────────────
    def show_startup_prompt(self):
        """Display setup instructions and wait for user confirmation."""
        console.print()
        console.rule(style="cyan")
        console.print(
            "[bold cyan]   ORBIT AI  ---  Personal Calibration[/bold cyan]"
        )
        console.rule(style="cyan")
        console.print()
        console.print("  [yellow]Before starting:[/yellow]")
        console.print(
            "  [white]1.[/white] Run [bold green]simulate_tgam.py[/bold green] "
            "in another terminal"
        )
        console.print(
            "     [dim]OR[/dim] connect your BioAmp hardware"
        )
        console.print(
            "  [white]2.[/white] Then press [bold]ENTER[/bold] to continue..."
        )
        console.print()
        input()  # wait for ENTER

    # ── CHANGE 4: Retry logic on connection failure ──────────────────────
    def connect(self):
        """Try connecting up to MAX_CONNECT_ATTEMPTS times with countdown."""
        for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
            # Create a fresh socket for each attempt
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            console.print(
                f"[cyan]Connecting to headset / simulator... "
                f"(attempt {attempt}/{MAX_CONNECT_ATTEMPTS})[/cyan]"
            )
            try:
                self.sock.connect(("127.0.0.1", 9999))
                self.sock.setblocking(False)
                console.print("[green]Connected successfully![/green]\n")
                return
            except Exception:
                self.sock.close()

                if attempt < MAX_CONNECT_ATTEMPTS:
                    for remaining in range(RETRY_DELAY_SEC, 0, -1):
                        console.print(
                            f"[yellow]  Retrying in {remaining} seconds... "
                            f"(attempt {attempt + 1}/{MAX_CONNECT_ATTEMPTS})[/yellow]",
                            end="\r",
                        )
                        time.sleep(1)
                    console.print()  # clear the \r line

        # ── CHANGE 2: Better connection error message ────────────────────
        console.print()
        console.print(
            "[bold red]Cannot connect to EEG stream on port 9999[/bold red]"
        )
        console.print()
        console.print("  [white]To fix this, open a NEW terminal and run:[/white]")
        console.print("    [bold green]python simulate_tgam.py[/bold green]")
        console.print()
        console.print("  [white]Then run calibrate.py again.[/white]")
        console.print()
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

    # ── CHANGE 3: Calibration prerequisites check ────────────────────────
    def check_signal_prerequisites(self):
        """Read a few packets and verify _raw_epoch data is available."""
        console.print("[cyan]Checking signal prerequisites...[/cyan]")

        packets_received = []
        deadline = time.time() + 10  # wait up to 10 seconds for packets
        while len(packets_received) < SIGNAL_CHECK_PACKETS and time.time() < deadline:
            packet = self.get_data()
            if packet is not None:
                packets_received.append(packet)
            time.sleep(0.1)

        has_raw_epoch = any("_raw_epoch" in p for p in packets_received)

        if not has_raw_epoch:
            console.print()
            console.print(
                "[bold yellow]Signal has no raw epoch data.[/bold yellow]"
            )
            console.print(
                "  This calibration requires [bold]simulate_tgam.py[/bold]"
            )
            console.print(
                "  in [green]Mode 1 (Local Clinical Data)[/green]."
            )
            console.print(
                "  The URL mode does not provide raw epochs."
            )
            console.print()
            answer = console.input(
                "[yellow]Continue anyway? (y/n): [/yellow]"
            ).strip().lower()
            if answer != "y":
                console.print("[dim]Exiting cleanly.[/dim]")
                raise SystemExit(0)
            console.print()
        else:
            console.print(
                f"[green]Signal OK — received {len(packets_received)} packets "
                f"with raw epoch data.[/green]\n"
            )

    def record_stage(self, stage_name, duration_sec=10):
        console.print(f"\n[bold yellow]Stage: {stage_name}[/bold yellow]")
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

    # ── CHANGE 5: Completion verification ────────────────────────────────
    def verify_and_display_profile(self, profile):
        """Load back the saved profile, verify keys, and display Rich table."""
        try:
            with open(PROFILE_PATH, "r") as f:
                loaded = json.load(f)
        except Exception as exc:
            console.print(f"[red]Could not reload profile for verification: {exc}[/red]")
            return

        expected_keys = [
            "baseline_alpha",
            "baseline_beta",
            "baseline_theta",
            "fatigue_threshold",
            "forward_beta_multiplier",
            "moabb_confidence_threshold",
            "calibration_date",
        ]
        missing = [k for k in expected_keys if k not in loaded]
        if missing:
            console.print(
                f"[bold red]Profile verification warning — missing keys: "
                f"{', '.join(missing)}[/bold red]"
            )

        table = Table(
            title="CALIBRATION PROFILE SAVED",
            title_style="bold green",
            border_style="bright_green",
            show_lines=True,
        )
        table.add_column("Metric", style="bold white", min_width=24)
        table.add_column("Value", style="green", justify="right", min_width=16)

        # Display the most important calibration values
        fatigue_t = loaded.get("fatigue_threshold", 0)
        beta_react = loaded.get("forward_beta_multiplier", 1)
        moabb_t = loaded.get("moabb_confidence_threshold", 0)
        cal_date = loaded.get("calibration_date", "N/A")

        table.add_row("Fatigue Threshold", f"{fatigue_t:.3f}")
        table.add_row("Beta Reactivity", f"+{((beta_react - 1) * 100):.1f}%")
        table.add_row("MOABB Threshold", f"{moabb_t:.2f}")
        table.add_row("Calibration Date", cal_date)
        table.add_row("Baseline Alpha", f"{loaded.get('baseline_alpha', 0):.4f}")
        table.add_row("Baseline Beta", f"{loaded.get('baseline_beta', 0):.4f}")
        table.add_row("Baseline Theta", f"{loaded.get('baseline_theta', 0):.4f}")

        console.print()
        console.print(table)
        console.print()
        console.print(
            "[bold green]You are ready to run predict_realtime.py[/bold green]"
        )
        console.print()

    def run(self):
        # CHANGE 1: Show startup prompt first
        self.show_startup_prompt()

        # CHANGE 4: Retry-enabled connection
        self.connect()

        # We drain the buffer so we have fresh data
        time.sleep(1)
        self.get_data()

        # CHANGE 3: Check signal prerequisites before recording
        self.check_signal_prerequisites()

        # Stages
        rest_packets = self.record_stage("BASELINE (Rest)", 15)
        fwd_packets  = self.record_stage("FORWARD (Motor Imagery)", 15)
        eyes_closed  = self.record_stage("DEEP REST (Eyes Closed)", 15)
        
        console.print("\n[cyan]Analyzing brainwaves and calculating thresholds...[/cyan]")
        
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

        # CHANGE 5: Verify and display profile
        self.verify_and_display_profile(profile)

if __name__ == "__main__":
    Calibrator().run()
