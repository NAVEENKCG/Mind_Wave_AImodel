import os
import csv
import time
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"

class SessionLogger:
    def __init__(self):
        self._ensure_log_dir()
        
        self.start_time = datetime.now()
        date_str = self.start_time.strftime("%Y-%m-%d_%H-%M")
        self.log_file = LOGS_DIR / f"session_{date_str}.csv"
        
        self.total_commands = 0
        self.cmd_counts = {"FORWARD": 0, "IDLE": 0, "STOP": 0}
        self.total_confidence = 0.0
        
        self.fatigue_events = 0
        self.signal_drops = 0
        
        self.in_fatigue = False
        self.in_signal_drop = False
        
        self._init_file()
        
    def _ensure_log_dir(self):
        if not LOGS_DIR.exists():
            LOGS_DIR.mkdir(parents=True)
            
    def _init_file(self):
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "command", "confidence", "signal_quality", 
                "fatigue_level", "attention", "meditation", "theta_beta_ratio"
            ])
            
    def log_step(self, command, confidence, signal_quality, fatigue_level, attention=0, meditation=0, theta_beta_ratio=0.0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, command, 
                f"{confidence:.4f}" if isinstance(confidence, (int, float)) else confidence, 
                signal_quality, fatigue_level, attention, meditation, 
                f"{theta_beta_ratio:.4f}" if isinstance(theta_beta_ratio, (int, float)) else theta_beta_ratio
            ])
            
        # Metrics Tracking
        cmd_upper = str(command).upper()
        if cmd_upper in self.cmd_counts:
            self.cmd_counts[cmd_upper] += 1
            self.total_commands += 1
            
            try:
                conf_val = float(confidence)
                # Convert 0-1 scale to percentage
                if conf_val <= 1.0:
                    conf_val *= 100
                self.total_confidence += conf_val
            except ValueError:
                pass
                
        # Fatigue events
        try:
            fatigue_val = float(fatigue_level)
            is_fatigued = fatigue_val > 0.8
        except ValueError:
            is_fatigued = str(fatigue_level).upper() in ["HIGH", "WARNING", "TRUE", "1"]
            
        if is_fatigued and not self.in_fatigue:
            self.fatigue_events += 1
            self.in_fatigue = True
        elif not is_fatigued:
            self.in_fatigue = False
            
        # Signal drops
        try:
            sq_val = float(signal_quality)
            is_dropped = sq_val == 0.0 or sq_val > 150 # Using common thresholds
        except ValueError:
            is_dropped = str(signal_quality).upper() in ["POOR", "DROP", "DISCONNECTED", "BAD", "0"]
            
        if is_dropped and not self.in_signal_drop:
            self.signal_drops += 1
            self.in_signal_drop = True
        elif not is_dropped:
            self.in_signal_drop = False
            
    def print_summary(self):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        seconds = int(duration.total_seconds())
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        duration_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        avg_conf = (self.total_confidence / self.total_commands) if self.total_commands > 0 else 0
        
        fwd_c = self.cmd_counts.get("FORWARD", 0)
        idl_c = self.cmd_counts.get("IDLE", 0)
        stp_c = self.cmd_counts.get("STOP", 0)
        
        fwd_p = (fwd_c / self.total_commands * 100) if self.total_commands > 0 else 0
        idl_p = (idl_c / self.total_commands * 100) if self.total_commands > 0 else 0
        stp_p = (stp_c / self.total_commands * 100) if self.total_commands > 0 else 0
        
        fatigue_str = f"{self.fatigue_events} warnings"
        drop_str = f"{self.signal_drops} time{' (recovered)' if self.signal_drops == 1 else 's (recovered)'}"
        
        summary = f"""  ═══════════════════════════════════
  SESSION SUMMARY
  ═══════════════════════════════════
  Duration:        {duration_str}
  Total Commands:  {self.total_commands}
  FORWARD:         {int(fwd_p):02d}% ({fwd_c} times)
  IDLE:            {int(idl_p):02d}% ({idl_c} times)
  STOP:            {int(stp_p):02d}% ({stp_c} times)
  Avg Confidence:  {avg_conf:.1f}%
  Fatigue Events:  {fatigue_str}
  Signal Drops:    {drop_str}
  ═══════════════════════════════════"""
        print(summary)

# For testing
if __name__ == "__main__":
    logger = SessionLogger()
    print(f"Logging to {logger.log_file}")
    logger.log_step("FORWARD", 0.814, "GOOD", "LOW", 80, 60, 1.2)
    time.sleep(0.1)
    logger.log_step("IDLE", 0.90, "GOOD", "HIGH", 40, 50, 0.8)
    time.sleep(0.1)
    logger.log_step("STOP", 0.75, "POOR", "LOW", 90, 80, 1.5)
    logger.print_summary()
