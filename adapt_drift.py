"""
ORBIT AI — Adaptive Neural Drift Engine (Model 2)
====================================================
Analyzes accumulated session logs to detect brain signal drift
(aging, medication changes, electrode degradation, neurodegeneration)
and produces an adaptation profile that refines Model 1's thresholds.

SAFETY GUARANTEES:
  1. NEVER opens, reads, or modifies moabb_csp_lda.pkl (Model 1)
  2. NEVER opens, reads, or modifies personal_profile.json
  3. ONLY writes to models/drift_profile.json
  4. All adapted values are clamped within hard-coded safety bounds
  5. If drift_profile.json is deleted, system reverts to original behavior

Usage:
  python adapt_drift.py              # Analyze logs and generate adaptation
  python adapt_drift.py --report     # Show drift report without saving
  python adapt_drift.py --reset      # Delete drift_profile.json (reset to defaults)
"""

import csv
import json
import time
import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

from config import (
    LOGS_DIR, MODELS_DIR,
    MIN_SESSIONS_FOR_ADAPTATION,
    MIN_EVENTS_PER_SESSION,
    DRIFT_EWMA_ALPHA,
    DRIFT_PROFILE_PATH,
)

# ══════════════════════════════════════════════════════════════════
# SAFETY BOUNDS — adaptation values are ALWAYS clamped to these.
# These are hard-coded, not configurable, to prevent dangerous drift.
# ══════════════════════════════════════════════════════════════════
SAFETY_BOUNDS = {
    "confidence_threshold_offset": (-0.15, 0.10),   # Can lower by 15%, raise by 10%
    "fatigue_threshold_scale":     (0.70, 1.50),     # 70%–150% of personal fatigue limit
    "signal_quality_floor_scale":  (0.60, 1.20),     # Signal floor can drop to 60%
    "biosensor_center_offset":     (-0.50, 0.50),    # BioSensor baseline shift ±0.5
    "warmup_extension_sec":        (0, 15),           # Add 0–15 extra warmup seconds
}


def clamp(value, bounds_key):
    """Clamp a value within its safety bounds."""
    lo, hi = SAFETY_BOUNDS[bounds_key]
    return max(lo, min(hi, value))


# ══════════════════════════════════════════════════════════════════
# SESSION LOG PARSER
# ══════════════════════════════════════════════════════════════════
def parse_session_csv(csv_path):
    """
    Parse a session CSV file and extract event data.
    Returns (session_id, events_list) or (None, []) on failure.

    Each event is a dict with keys:
      timestamp, command, confidence, signal_quality, fatigue_state,
      elapsed_seconds, raw_forward_prob, signal_rms,
      theta_power, alpha_power, beta_power
    """
    events = []
    session_id = csv_path.stem.replace("session_", "")

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip summary comment lines
                if not row.get("timestamp") or row["timestamp"].startswith("#"):
                    continue
                try:
                    event = {
                        "timestamp": row.get("timestamp", ""),
                        "command": row.get("command", "IDLE"),
                        "confidence": float(row.get("confidence", 0)),
                        "signal_quality": row.get("signal_quality", "UNKNOWN"),
                        "fatigue_state": row.get("fatigue_state", "NORMAL"),
                        "elapsed_seconds": float(row.get("elapsed_seconds", 0)),
                        # New columns (may be absent in older logs)
                        "raw_forward_prob": float(row.get("raw_forward_prob", 0)) if row.get("raw_forward_prob") else None,
                        "signal_rms": float(row.get("signal_rms", 0)) if row.get("signal_rms") else None,
                        "theta_power": float(row.get("theta_power", 0)) if row.get("theta_power") else None,
                        "alpha_power": float(row.get("alpha_power", 0)) if row.get("alpha_power") else None,
                        "beta_power": float(row.get("beta_power", 0)) if row.get("beta_power") else None,
                    }
                    events.append(event)
                except (ValueError, TypeError):
                    continue
    except Exception:
        return None, []

    return session_id, events


def parse_session_summary(json_path):
    """Parse a session summary JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# DRIFT ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════
class DriftAnalyzer:
    """
    Analyzes session logs to detect brain signal drift and compute
    threshold adaptations.

    IMPORTANT: This class NEVER accesses moabb_csp_lda.pkl or
    personal_profile.json. It only reads from logs/ and writes
    to models/drift_profile.json.
    """

    def __init__(self):
        self.sessions = []  # List of (session_id, timestamp, metrics_dict)
        self.ewma_alpha = DRIFT_EWMA_ALPHA

    def load_sessions(self):
        """Scan logs/ directory and load all valid sessions."""
        logs_dir = LOGS_DIR
        if not logs_dir.exists():
            return 0

        csv_files = sorted(logs_dir.glob("session_*.csv"))
        loaded = 0

        for csv_path in csv_files:
            # Skip summary JSON files
            if "_summary" in csv_path.name:
                continue

            session_id, events = parse_session_csv(csv_path)
            if session_id is None or len(events) < MIN_EVENTS_PER_SESSION:
                continue

            # Extract session-level metrics
            confidences = [e["confidence"] for e in events if e["confidence"] > 0]
            forward_count = sum(1 for e in events if e["command"] == "FORWARD")
            idle_count = sum(1 for e in events if e["command"] == "IDLE")
            total = len(events)

            # Signal RMS values (may be None in older logs)
            rms_values = [e["signal_rms"] for e in events if e["signal_rms"] is not None]

            # Raw forward probabilities (may be None in older logs)
            raw_probs = [e["raw_forward_prob"] for e in events if e["raw_forward_prob"] is not None]

            # Band powers
            theta_vals = [e["theta_power"] for e in events if e["theta_power"] is not None]
            alpha_vals = [e["alpha_power"] for e in events if e["alpha_power"] is not None]
            beta_vals = [e["beta_power"] for e in events if e["beta_power"] is not None]

            # Fatigue events
            fatigue_events = sum(1 for e in events if e["fatigue_state"] in ("WARNING", "CRITICAL", "ALERT"))

            # Parse timestamp from session ID (format: YYYYMMDD_HHMMSS)
            try:
                ts = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
            except ValueError:
                try:
                    # Older format: YYYY-MM-DD
                    ts = datetime.strptime(session_id, "%Y-%m-%d")
                except ValueError:
                    ts = datetime.now()

            metrics = {
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "median_confidence": float(np.median(confidences)) if confidences else 0.0,
                "forward_ratio": forward_count / total if total > 0 else 0.0,
                "idle_ratio": idle_count / total if total > 0 else 0.0,
                "total_events": total,
                "fatigue_event_ratio": fatigue_events / total if total > 0 else 0.0,
                "avg_signal_rms": float(np.mean(rms_values)) if rms_values else None,
                "avg_raw_forward_prob": float(np.mean(raw_probs)) if raw_probs else None,
                "avg_theta": float(np.mean(theta_vals)) if theta_vals else None,
                "avg_alpha": float(np.mean(alpha_vals)) if alpha_vals else None,
                "avg_beta": float(np.mean(beta_vals)) if beta_vals else None,
                # Session duration from last event
                "duration_sec": events[-1]["elapsed_seconds"] if events else 0,
            }

            self.sessions.append((session_id, ts, metrics))
            loaded += 1

        # Sort by timestamp (oldest first)
        self.sessions.sort(key=lambda x: x[1])
        return loaded

    def compute_trend(self, values):
        """
        Compute linear regression slope on a series of values.
        Returns (slope, r_squared) indicating direction and strength.
        """
        if len(values) < 3:
            return 0.0, 0.0

        x = np.arange(len(values), dtype=np.float64)
        y = np.array(values, dtype=np.float64)

        # Remove NaN/inf values
        mask = np.isfinite(y)
        if mask.sum() < 3:
            return 0.0, 0.0

        x = x[mask]
        y = y[mask]

        # Linear regression: y = slope * x + intercept
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-12:
            return 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))

        return float(slope), float(max(0, r_squared))

    def compute_ewma(self, values):
        """
        Compute Exponentially Weighted Moving Average.
        Recent values weighted more heavily than older ones.
        """
        if not values:
            return 0.0

        ewma = values[0]
        for v in values[1:]:
            ewma = self.ewma_alpha * v + (1 - self.ewma_alpha) * ewma
        return float(ewma)

    def analyze(self):
        """
        Run full drift analysis and return adaptation dict.
        Returns (adaptations, report) tuple.
        """
        if len(self.sessions) < MIN_SESSIONS_FOR_ADAPTATION:
            return None, {
                "status": "insufficient_data",
                "sessions_found": len(self.sessions),
                "sessions_required": MIN_SESSIONS_FOR_ADAPTATION,
            }

        # ── Extract time-series metrics ─────────────────────────────
        confidences = [m["avg_confidence"] for _, _, m in self.sessions]
        forward_ratios = [m["forward_ratio"] for _, _, m in self.sessions]
        fatigue_ratios = [m["fatigue_event_ratio"] for _, _, m in self.sessions]

        # Signal RMS (may be absent in older logs)
        rms_values = [m["avg_signal_rms"] for _, _, m in self.sessions if m["avg_signal_rms"] is not None]

        # ── Trend analysis ──────────────────────────────────────────
        conf_slope, conf_r2 = self.compute_trend(confidences)
        fwd_slope, fwd_r2 = self.compute_trend(forward_ratios)
        fatigue_slope, fatigue_r2 = self.compute_trend(fatigue_ratios)
        rms_slope, rms_r2 = self.compute_trend(rms_values) if rms_values else (0.0, 0.0)

        # ── EWMA of recent sessions vs early sessions ───────────────
        n = len(self.sessions)
        early_n = max(1, n // 3)
        late_n = max(1, n // 3)

        early_conf = np.mean(confidences[:early_n])
        late_conf = self.compute_ewma(confidences[-late_n:])
        conf_change = late_conf - early_conf  # Negative = degrading

        early_fwd = np.mean(forward_ratios[:early_n])
        late_fwd = self.compute_ewma(forward_ratios[-late_n:])

        # ── Classify drift severity ─────────────────────────────────
        if conf_change < -0.15:
            conf_trend = "declining_severe"
        elif conf_change < -0.05:
            conf_trend = "declining"
        elif conf_change > 0.05:
            conf_trend = "improving"
        else:
            conf_trend = "stable"

        signal_degradation = (rms_slope < -0.01 and rms_r2 > 0.3) if rms_values else False

        # ── Compute adaptations ─────────────────────────────────────
        adaptations = {}

        # 1. Confidence threshold offset
        # If confidence is declining, lower the threshold to compensate
        if conf_trend in ("declining", "declining_severe"):
            # Scale offset by severity: -0.05 for mild, up to -0.15 for severe
            raw_offset = conf_change * 0.5  # Half the confidence drop as threshold reduction
            adaptations["confidence_threshold_offset"] = clamp(raw_offset, "confidence_threshold_offset")
        else:
            adaptations["confidence_threshold_offset"] = 0.0

        # 2. Fatigue threshold scaling
        # If fatigue events are increasing, relax the threshold
        if fatigue_slope > 0.005 and fatigue_r2 > 0.2:
            scale = 1.0 + min(0.5, fatigue_slope * 10)
            adaptations["fatigue_threshold_scale"] = clamp(scale, "fatigue_threshold_scale")
        else:
            adaptations["fatigue_threshold_scale"] = 1.0

        # 3. Signal quality floor
        # If signal RMS is declining, lower the floor to keep accepting data
        if signal_degradation:
            scale = max(0.6, 1.0 + rms_slope * 5)
            adaptations["signal_quality_floor_scale"] = clamp(scale, "signal_quality_floor_scale")
        else:
            adaptations["signal_quality_floor_scale"] = 1.0

        # 4. BioSensor center offset
        # If the beta/alpha ratio baseline is shifting, track it
        if len(rms_values) > 3:
            recent_rms = self.compute_ewma(rms_values[-5:]) if len(rms_values) >= 5 else np.mean(rms_values)
            early_rms = np.mean(rms_values[:early_n])
            rms_shift = (recent_rms - early_rms) / (early_rms + 1e-12)
            adaptations["biosensor_center_offset"] = clamp(rms_shift * 0.3, "biosensor_center_offset")
        else:
            adaptations["biosensor_center_offset"] = 0.0

        # 5. Warmup extension
        # If early-session accuracy is worse than later, extend warmup
        if n >= 5:
            early_session_confs = []
            late_session_confs = []
            for _, _, m in self.sessions[-5:]:
                # We'd need per-event timing, approximate with overall
                early_session_confs.append(m["avg_confidence"])

            # If recent sessions start weak, add warmup
            if late_conf < 0.6:
                extra_warmup = int((0.7 - late_conf) * 30)  # Up to ~15s
                adaptations["warmup_extension_sec"] = clamp(extra_warmup, "warmup_extension_sec")
            else:
                adaptations["warmup_extension_sec"] = 0
        else:
            adaptations["warmup_extension_sec"] = 0

        # ── Build report ────────────────────────────────────────────
        recommend_recalibration = (
            conf_trend == "declining_severe"
            or signal_degradation
            or late_conf < 0.50
        )

        report = {
            "status": "adaptation_computed",
            "confidence_trend": conf_trend,
            "confidence_slope": round(conf_slope, 6),
            "confidence_r_squared": round(conf_r2, 4),
            "avg_confidence_early": round(float(early_conf * 100), 1),
            "avg_confidence_recent": round(float(late_conf * 100), 1),
            "forward_ratio_early": round(float(early_fwd), 3),
            "forward_ratio_recent": round(float(late_fwd), 3),
            "fatigue_trend_slope": round(fatigue_slope, 6),
            "signal_degradation_detected": signal_degradation,
            "signal_rms_sessions_available": len(rms_values),
            "recommended_recalibration": recommend_recalibration,
        }

        return adaptations, report

    def save_profile(self, adaptations, report):
        """
        Save adaptation profile to drift_profile.json.
        NEVER touches moabb_csp_lda.pkl or personal_profile.json.
        """
        # Check if a previous profile exists to track version
        version = 1
        if DRIFT_PROFILE_PATH.exists():
            try:
                with open(DRIFT_PROFILE_PATH, "r") as f:
                    old = json.load(f)
                    version = old.get("adaptation_version", 0) + 1
            except Exception:
                pass

        profile = {
            "adaptation_version": version,
            "adaptation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sessions_analyzed": len(self.sessions),
            "total_events_analyzed": sum(m["total_events"] for _, _, m in self.sessions),
            "adaptations": {k: round(v, 6) for k, v in adaptations.items()},
            "drift_report": report,
            "safety_bounds": {k: list(v) for k, v in SAFETY_BOUNDS.items()},
            "_note": "This file is generated by adapt_drift.py. It NEVER modifies moabb_csp_lda.pkl (Model 1). Delete this file to reset all adaptations."
        }

        MODELS_DIR.mkdir(exist_ok=True)
        with open(DRIFT_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)

        return profile

    def display_report(self, adaptations, report, saved=False):
        """Display the drift analysis results."""
        if HAS_RICH:
            self._display_rich(adaptations, report, saved)
        else:
            self._display_plain(adaptations, report, saved)

    def _display_rich(self, adaptations, report, saved):
        console.print()
        console.rule("[bold cyan]ORBIT AI — Adaptive Neural Drift Engine[/bold cyan]")
        console.print()

        # ── Status ─────────────────────────────────────────────────
        if report.get("status") == "insufficient_data":
            console.print(
                f"[yellow]Insufficient data: {report['sessions_found']} sessions found, "
                f"need {report['sessions_required']}.[/yellow]"
            )
            console.print("[dim]Run more BCI sessions (with >100 events each) to enable adaptation.[/dim]")
            console.print()
            return

        # ── Drift Report Table ─────────────────────────────────────
        trend_table = Table(
            title="Signal Drift Analysis",
            title_style="bold white",
            border_style="cyan",
            show_lines=True,
        )
        trend_table.add_column("Metric", style="bold white", min_width=28)
        trend_table.add_column("Value", style="green", justify="right", min_width=18)

        conf_trend = report["confidence_trend"]
        trend_color = "red" if "declining" in conf_trend else "green" if conf_trend == "improving" else "yellow"
        trend_table.add_row("Sessions Analyzed", str(len(self.sessions)))
        trend_table.add_row(
            "Confidence Trend",
            f"[{trend_color}]{conf_trend.upper()}[/{trend_color}]"
        )
        trend_table.add_row(
            "Avg Confidence (early)",
            f"{report['avg_confidence_early']:.1f}%"
        )
        trend_table.add_row(
            "Avg Confidence (recent)",
            f"{report['avg_confidence_recent']:.1f}%"
        )
        trend_table.add_row(
            "Signal Degradation",
            "[red]DETECTED[/red]" if report["signal_degradation_detected"] else "[green]NONE[/green]"
        )
        trend_table.add_row(
            "Recalibration Recommended",
            "[red bold]YES[/red bold]" if report["recommended_recalibration"] else "[green]No[/green]"
        )

        console.print(trend_table)
        console.print()

        # ── Adaptations Table ──────────────────────────────────────
        adapt_table = Table(
            title="Computed Adaptations (Safety-Bounded)",
            title_style="bold white",
            border_style="bright_green",
            show_lines=True,
        )
        adapt_table.add_column("Parameter", style="bold white", min_width=30)
        adapt_table.add_column("Value", style="cyan", justify="right", min_width=12)
        adapt_table.add_column("Bounds", style="dim", justify="right", min_width=16)

        for key, val in adaptations.items():
            bounds = SAFETY_BOUNDS.get(key, (None, None))
            bounds_str = f"[{bounds[0]}, {bounds[1]}]" if bounds[0] is not None else "—"

            if key == "confidence_threshold_offset":
                display = f"{val:+.3f}" if val != 0 else "0 (no change)"
            elif key == "fatigue_threshold_scale":
                display = f"×{val:.2f}" if val != 1.0 else "×1.00 (no change)"
            elif key == "signal_quality_floor_scale":
                display = f"×{val:.2f}" if val != 1.0 else "×1.00 (no change)"
            elif key == "biosensor_center_offset":
                display = f"{val:+.3f}" if val != 0 else "0 (no change)"
            elif key == "warmup_extension_sec":
                display = f"+{int(val)}s" if val > 0 else "0s (no change)"
            else:
                display = str(val)

            adapt_table.add_row(key, display, bounds_str)

        console.print(adapt_table)
        console.print()

        if saved:
            console.print(f"[bold green]Profile saved to: models/drift_profile.json[/bold green]")
            console.print("[dim]This file is loaded automatically by predict_realtime.py and predict_websocket.py[/dim]")
        else:
            console.print("[yellow]Report only mode — no changes saved.[/yellow]")
            console.print("[dim]Run without --report flag to save the adaptation.[/dim]")

        if report.get("recommended_recalibration"):
            console.print()
            console.print(
                "[bold red]⚠ Signal drift is significant. "
                "Consider running 'python calibrate.py' for a fresh personal profile.[/bold red]"
            )

        console.print()
        console.print("[dim]NOTE: The original model (moabb_csp_lda.pkl) was NOT modified.[/dim]")
        console.print()

    def _display_plain(self, adaptations, report, saved):
        print("\n" + "=" * 50)
        print("ORBIT AI — Adaptive Neural Drift Engine")
        print("=" * 50)

        if report.get("status") == "insufficient_data":
            print(f"Insufficient data: {report['sessions_found']} sessions, need {report['sessions_required']}.")
            return

        print(f"\nSessions analyzed: {len(self.sessions)}")
        print(f"Confidence trend: {report['confidence_trend']}")
        print(f"Early confidence: {report['avg_confidence_early']:.1f}%")
        print(f"Recent confidence: {report['avg_confidence_recent']:.1f}%")
        print(f"Signal degradation: {'DETECTED' if report['signal_degradation_detected'] else 'None'}")
        print(f"\nAdaptations:")
        for key, val in adaptations.items():
            print(f"  {key}: {val}")
        print(f"\n{'Profile saved.' if saved else 'Report only — not saved.'}")
        print(f"NOTE: moabb_csp_lda.pkl was NOT modified.\n")


# ══════════════════════════════════════════════════════════════════
# UTILITY: Load drift profile (used by prediction scripts)
# ══════════════════════════════════════════════════════════════════
def load_drift_profile():
    """
    Load the drift adaptation profile from disk.
    Returns the profile dict, or None if not found / disabled.

    This function is imported by predict_realtime.py and
    predict_websocket.py.
    """
    try:
        from config import ADAPTATION_ENABLED
        if not ADAPTATION_ENABLED:
            return None
    except ImportError:
        pass

    if not DRIFT_PROFILE_PATH.exists():
        return None

    try:
        with open(DRIFT_PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # Validate structure
        if "adaptations" not in profile:
            return None

        return profile
    except Exception:
        return None


def apply_drift_to_threshold(base_threshold, drift_profile):
    """
    Apply drift offset to a confidence threshold.
    Returns the adapted threshold, clamped to absolute safety limits.

    Args:
        base_threshold: The original threshold (e.g., 0.60)
        drift_profile: The loaded drift profile dict (or None)

    Returns:
        Adapted threshold value (float)
    """
    if drift_profile is None:
        return base_threshold

    offset = drift_profile.get("adaptations", {}).get("confidence_threshold_offset", 0.0)
    adapted = base_threshold + offset

    # Absolute safety: threshold can never go below 0.35 or above 0.85
    return max(0.35, min(0.85, adapted))


def apply_drift_to_fatigue(base_fatigue_threshold, drift_profile):
    """
    Apply drift scaling to a fatigue threshold.

    Args:
        base_fatigue_threshold: The original fatigue threshold
        drift_profile: The loaded drift profile dict (or None)

    Returns:
        Adapted fatigue threshold value (float)
    """
    if drift_profile is None:
        return base_fatigue_threshold

    scale = drift_profile.get("adaptations", {}).get("fatigue_threshold_scale", 1.0)
    return base_fatigue_threshold * scale


def get_warmup_extension(drift_profile):
    """Get extra warmup seconds from drift profile."""
    if drift_profile is None:
        return 0
    return int(drift_profile.get("adaptations", {}).get("warmup_extension_sec", 0))


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ORBIT AI — Adaptive Neural Drift Engine (Model 2)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Show drift report without saving changes"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete drift_profile.json and reset to original behavior"
    )
    args = parser.parse_args()

    # ── Reset mode ──────────────────────────────────────────────
    if args.reset:
        if DRIFT_PROFILE_PATH.exists():
            DRIFT_PROFILE_PATH.unlink()
            if HAS_RICH:
                console.print("[green]Drift profile deleted. System reverted to original behavior.[/green]")
            else:
                print("Drift profile deleted. System reverted to original behavior.")
        else:
            if HAS_RICH:
                console.print("[yellow]No drift profile found — already using original behavior.[/yellow]")
            else:
                print("No drift profile found — already using original behavior.")
        return

    # ── Analyze ─────────────────────────────────────────────────
    analyzer = DriftAnalyzer()
    n_loaded = analyzer.load_sessions()

    if HAS_RICH:
        console.print(f"[cyan]Loaded {n_loaded} valid sessions from logs/[/cyan]")
    else:
        print(f"Loaded {n_loaded} valid sessions from logs/")

    adaptations, report = analyzer.analyze()

    if adaptations is None:
        analyzer.display_report({}, report, saved=False)
        return

    if args.report:
        analyzer.display_report(adaptations, report, saved=False)
    else:
        analyzer.save_profile(adaptations, report)
        analyzer.display_report(adaptations, report, saved=True)


if __name__ == "__main__":
    main()
