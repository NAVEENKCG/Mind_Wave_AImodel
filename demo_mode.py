"""
ORBIT AI — One-Command Demo Launcher
=====================================
Launches the EEG simulator and BCI dashboard together:
    python demo_mode.py

Handles dependency checks, model verification, process
lifecycle, crash recovery, and clean shutdown on Ctrl+C.
"""

import importlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── Rich console (with graceful fallback) ────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    # If Rich itself is missing we still need to report it
    print("ERROR: 'rich' package is not installed.")
    print("Run:  pip install rich")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MOABB_MODEL = MODELS_DIR / "moabb_csp_lda.pkl"

# Use 'python' on Windows, 'python3' on Unix unless overridden
PYTHON = sys.executable or ("python" if platform.system() == "Windows" else "python3")

SIM_SCRIPT = str(ROOT / "simulate_tgam.py")
PRED_SCRIPT = str(ROOT / "predict_realtime.py")

SIM_HOST = "127.0.0.1"
SIM_PORT = 9999
SIM_READY_TIMEOUT = 10  # seconds


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — Dependency Check
# ─────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "torch":   "torch",
    "numpy":   "numpy",
    "rich":    "rich",
    "scipy":   "scipy",
    "sklearn": "scikit-learn",   # pip name differs from import name
    "socket":  None,             # stdlib, always present
}


def check_dependencies() -> bool:
    """Return True if all required packages are importable."""
    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if pip_name is None:
            continue  # stdlib — skip
        try:
            importlib.import_module(import_name)
        except ImportError:
            console.print(f"[bold red]Missing: {import_name}[/bold red]")
            console.print(f"  Run: [green]pip install {pip_name}[/green]")
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — Model Check
# ─────────────────────────────────────────────────────────────────────────
def check_model() -> bool:
    """Return True if we should continue (model found OR user chose yes)."""
    if MOABB_MODEL.exists():
        console.print("[green]Trained model found.[/green]")
        return True

    console.print("[bold yellow]No trained model found.[/bold yellow]")
    console.print("  Run [green]python train_moabb.py[/green] first for best results.")
    answer = console.input(
        "[yellow]Continue in BioSensor-only mode? (y/n): [/yellow]"
    ).strip().lower()
    return answer == "y"


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — Launch Simulator
# ─────────────────────────────────────────────────────────────────────────
def launch_simulator() -> subprocess.Popen:
    """Start simulate_tgam.py in a background subprocess."""
    console.print("\n[cyan]Launching EEG simulator...[/cyan]")

    creation_flags = 0
    if platform.system() == "Windows":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [PYTHON, SIM_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )

    # Give the process a moment to initialise
    time.sleep(3)

    if proc.poll() is not None:
        console.print("[bold red]Simulator exited immediately (bad script?).[/bold red]")
        sys.exit(1)

    console.print(f"[green]Simulator started (PID: {proc.pid})[/green]")
    return proc


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — Wait for Simulator TCP Port
# ─────────────────────────────────────────────────────────────────────────
def wait_for_simulator(proc: subprocess.Popen) -> bool:
    """Block until the simulator's TCP port is accepting connections."""
    for remaining in range(SIM_READY_TIMEOUT, 0, -1):
        # Check the process hasn't died
        if proc.poll() is not None:
            console.print("[bold red]Simulator process exited unexpectedly.[/bold red]")
            return False

        try:
            with socket.create_connection((SIM_HOST, SIM_PORT), timeout=1):
                console.print("[green]Simulator is ready.[/green]")
                return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            console.print(
                f"[dim]Waiting for simulator... {remaining}s[/dim]",
                end="\r",
            )
            time.sleep(1)

    console.print()
    console.print("[bold red]Simulator failed to start within timeout.[/bold red]")
    return False


# ─────────────────────────────────────────────────────────────────────────
# STEP 5 — Launch Dashboard
# ─────────────────────────────────────────────────────────────────────────
def launch_dashboard() -> subprocess.Popen:
    """Start predict_realtime.py --demo."""
    console.print("\n[cyan]Launching BCI dashboard (demo mode)...[/cyan]")

    proc = subprocess.Popen(
        [PYTHON, PRED_SCRIPT, "--demo"],
    )

    console.print(f"[green]Dashboard started (PID: {proc.pid})[/green]")
    return proc


# ─────────────────────────────────────────────────────────────────────────
# STEP 6 — Demo Instructions
# ─────────────────────────────────────────────────────────────────────────
def show_instructions() -> None:
    """Print the demo-controls panel."""
    instructions = (
        "[bold white]In the SIMULATOR window:[/bold white]\n"
        "  Press [bold green]\\[1][/bold green]  ->  Show FORWARD brain state\n"
        "  Press [bold green]\\[0][/bold green]  ->  Show IDLE brain state\n"
        "\n"
        "[bold white]The DASHBOARD will show the result.[/bold white]\n"
        "\n"
        "[dim]Press Ctrl+C here to stop everything.[/dim]"
    )
    console.print()
    console.print(Panel(
        instructions,
        title="[bold cyan]ORBIT AI  --  LIVE DEMO[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


# ─────────────────────────────────────────────────────────────────────────
# STEP 7 — Keep Alive / Crash Recovery / Clean Shutdown
# ─────────────────────────────────────────────────────────────────────────
def terminate_safely(proc: subprocess.Popen, name: str) -> None:
    """Terminate a subprocess, falling back to kill after 5 s."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    console.print(f"[dim]{name} stopped.[/dim]")


def run_demo() -> None:
    """Orchestrate the full demo lifecycle."""

    # ── STEP 1 ────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]ORBIT AI  --  Demo Launcher[/bold cyan]")
    console.print()

    if not check_dependencies():
        console.print("\n[bold red]Install missing packages and try again.[/bold red]")
        sys.exit(1)
    console.print("[green]All dependencies OK.[/green]")

    # ── STEP 2 ────────────────────────────────────────────────────────
    if not check_model():
        console.print("[dim]Exiting.[/dim]")
        sys.exit(0)

    # ── STEP 3 ────────────────────────────────────────────────────────
    proc_sim = launch_simulator()

    # ── STEP 4 ────────────────────────────────────────────────────────
    if not wait_for_simulator(proc_sim):
        terminate_safely(proc_sim, "Simulator")
        sys.exit(1)

    # ── STEP 5 ────────────────────────────────────────────────────────
    proc_pred = launch_dashboard()

    # ── STEP 6 ────────────────────────────────────────────────────────
    show_instructions()

    # ── STEP 7 — Keep-alive loop ─────────────────────────────────────
    try:
        while True:
            time.sleep(1)

            # Simulator crash → restart
            if proc_sim.poll() is not None:
                console.print("[yellow]Simulator crashed. Restarting...[/yellow]")
                proc_sim = launch_simulator()
                if not wait_for_simulator(proc_sim):
                    console.print("[bold red]Restart failed. Shutting down.[/bold red]")
                    break

            # Dashboard exited → stop everything
            if proc_pred.poll() is not None:
                console.print("[yellow]Dashboard exited.[/yellow]")
                break

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping ORBIT AI Demo...[/bold red]")

    finally:
        terminate_safely(proc_pred, "Dashboard")
        terminate_safely(proc_sim, "Simulator")
        console.print("[green]All processes stopped cleanly.[/green]\n")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_demo()
