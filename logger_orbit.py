"""
ORBIT AI — Session Logger
Tracks BCI session events to CSV with Rich console summaries.
"""

import csv
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Internal logger (file/stderr, never crashes caller) ──────────────────
_log = logging.getLogger("orbit_logger")
_log.setLevel(logging.WARNING)
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[OrbitLogger] %(levelname)s: %(message)s"))
    _log.addHandler(_handler)


class OrbitLogger:
    """Thread-safe CSV session logger for ORBIT AI BCI pipeline."""

    # Commands tracked in the summary breakdown
    _TRACKED_COMMANDS = ("IDLE", "FORWARD", "LEFT", "RIGHT", "PUSH", "PULL")

    def __init__(self) -> None:
        self.session_id: str = ""
        self.session_start_time: float = 0.0
        self.log_dir: Path = ROOT / "logs"
        self.session_file: Path = Path("")  # set on log_session_start
        self.event_count: int = 0
        self.command_counts: dict[str, int] = {cmd: 0 for cmd in self._TRACKED_COMMANDS}
        self._confidences: list[float] = []
        self._lock = threading.Lock()
        self._console = Console() if Console else None

    # ── 1. Session start ─────────────────────────────────────────────────
    def log_session_start(self) -> None:
        """Create logs/ dir, open a new CSV, and write the header row."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log.error("Cannot create log directory: %s", exc)
            return

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        self.event_count = 0
        self.command_counts = {cmd: 0 for cmd in self._TRACKED_COMMANDS}
        self._confidences.clear()

        self.session_file = self.log_dir / f"session_{self.session_id}.csv"

        try:
            with self._lock:
                with open(self.session_file, mode="w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        "timestamp",
                        "event_id",
                        "command",
                        "confidence",
                        "signal_quality",
                        "fatigue_state",
                        "elapsed_seconds",
                    ])
        except OSError as exc:
            _log.error("Cannot write CSV header: %s", exc)
            return

        msg = f"Session logging started -> logs/session_{self.session_id}.csv"
        if self._console:
            self._console.print(f"[bold green]:memo: {msg}[/bold green]")
        else:
            print(msg)

    # ── 2. Log event ─────────────────────────────────────────────────────
    def log_event(
        self,
        command: str,
        confidence: float,
        signal_quality: str,
        fatigue_state: str,
    ) -> None:
        """Append one data row to the session CSV."""
        elapsed = time.time() - self.session_start_time if self.session_start_time else 0.0

        try:
            with self._lock:
                self.event_count += 1
                with open(self.session_file, mode="a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        datetime.now().isoformat(),
                        self.event_count,
                        command,
                        f"{confidence:.4f}",
                        signal_quality,
                        fatigue_state,
                        f"{elapsed:.2f}",
                    ])
        except OSError as exc:
            _log.error("Cannot write event row: %s", exc)
            return

        # Update counters (outside the file-write lock scope is fine;
        # still inside the GIL, and we only mutate from here).
        if command in self.command_counts:
            self.command_counts[command] += 1
        else:
            self.command_counts[command] = 1

        try:
            self._confidences.append(float(confidence))
        except (TypeError, ValueError):
            pass

    # ── 3. Session end ───────────────────────────────────────────────────
    def log_session_end(self) -> None:
        """Write summary block to CSV, print Rich table, save JSON."""
        summary = self.get_summary()
        duration_sec = summary["duration_seconds"]
        minutes, seconds = divmod(int(duration_sec), 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"
        avg_conf = summary["average_confidence"]

        # ── Append summary comment block to CSV ──────────────────────────
        try:
            with self._lock:
                with open(self.session_file, mode="a", newline="", encoding="utf-8") as fh:
                    fh.write(f"\n# SESSION SUMMARY\n")
                    fh.write(f"# Total Events: {summary['total_events']}\n")
                    fh.write(f"# Duration: {duration_str}\n")
                    for cmd in self._TRACKED_COMMANDS:
                        fh.write(f"# {cmd} count: {summary['command_counts'].get(cmd, 0)}\n")
                    fh.write(f"# Average Confidence: {avg_conf:.2f}%\n")
        except OSError as exc:
            _log.error("Cannot write session summary to CSV: %s", exc)

        # ── Rich console table ───────────────────────────────────────────
        if self._console and Table:
            table = Table(
                title="🧠 ORBIT AI — Session Summary",
                title_style="bold cyan",
                border_style="bright_blue",
                show_lines=True,
            )
            table.add_column("Metric", style="bold white", min_width=22)
            table.add_column("Value", style="green", justify="right", min_width=14)

            table.add_row("Session ID", self.session_id)
            table.add_row("Total Events", str(summary["total_events"]))
            table.add_row("Duration", duration_str)
            for cmd in self._TRACKED_COMMANDS:
                table.add_row(f"{cmd} count", str(summary["command_counts"].get(cmd, 0)))
            table.add_row("Avg Confidence", f"{avg_conf:.2f}%")

            self._console.print()
            self._console.print(table)
            self._console.print()
        else:
            # Fallback plain-text summary
            print(f"\n{'=' * 40}")
            print("ORBIT AI — Session Summary")
            print(f"{'=' * 40}")
            print(f"  Session ID      : {self.session_id}")
            print(f"  Total Events    : {summary['total_events']}")
            print(f"  Duration        : {duration_str}")
            for cmd in self._TRACKED_COMMANDS:
                print(f"  {cmd:15s} : {summary['command_counts'].get(cmd, 0)}")
            print(f"  Avg Confidence  : {avg_conf:.2f}%")
            print(f"{'=' * 40}\n")

        # ── Save JSON summary ────────────────────────────────────────────
        json_path = self.log_dir / f"session_{self.session_id}_summary.json"
        try:
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(summary, jf, indent=2)
        except OSError as exc:
            _log.error("Cannot write summary JSON: %s", exc)

    # ── 4. Get summary ───────────────────────────────────────────────────
    def get_summary(self) -> dict:
        """Return current session statistics as a dictionary."""
        duration = time.time() - self.session_start_time if self.session_start_time else 0.0
        avg_conf = (
            (sum(self._confidences) / len(self._confidences) * 100)
            if self._confidences
            else 0.0
        )
        return {
            "session_id": self.session_id,
            "total_events": self.event_count,
            "duration_seconds": round(duration, 2),
            "command_counts": dict(self.command_counts),
            "average_confidence": round(avg_conf, 2),
        }


# ── Global instance (imported by predict_realtime.py / predict_websocket.py) ─
logger = OrbitLogger()

if __name__ == "__main__":
    # Quick self-test
    logger.log_session_start()
    logger.log_event("FORWARD", 0.9512, "STRONG", "LOW")
    logger.log_event("IDLE", 0.3201, "WEAK", "MEDIUM")
    logger.log_event("FORWARD", 0.8877, "STRONG", "LOW")
    logger.log_session_end()
