import csv
import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"

class OrbitLogger:
    def __init__(self):
        self._ensure_log_dir()
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = LOGS_DIR / f"session_{date_str}.csv"
        self._init_file()
        
    def _ensure_log_dir(self):
        if not LOGS_DIR.exists():
            LOGS_DIR.mkdir(parents=True)
            
    def _init_file(self):
        # Write header if file does not exist
        if not self.log_file.exists():
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "command", 
                    "confidence", 
                    "signal_quality", 
                    "fatigue_level", 
                    "attention", 
                    "meditation", 
                    "theta_beta_ratio"
                ])
                
    def log_event(self, command, confidence, signal_quality, fatigue_level, attention=0, meditation=0, theta_beta_ratio=0.0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                command,
                f"{confidence:.2f}" if isinstance(confidence, (int, float)) else confidence,
                signal_quality,
                fatigue_level,
                attention,
                meditation,
                f"{theta_beta_ratio:.2f}" if isinstance(theta_beta_ratio, (int, float)) else theta_beta_ratio
            ])
            
    def log_session_start(self):
        self.log_event("SESSION_START", 1.0, "N/A", "N/A")
        
    def log_session_end(self):
        self.log_event("SESSION_END", 1.0, "N/A", "N/A")
        
    def log_alert(self, alert_message):
        self.log_event(f"ALERT: {alert_message}", 0.0, "N/A", "N/A")

# Global logger instance
logger = OrbitLogger()

if __name__ == "__main__":
    # Test the logger
    logger.log_session_start()
    logger.log_event("FORWARD", 0.95, "STRONG", "LOW", 80, 60, 1.2)
    logger.log_alert("Poor signal detected")
    logger.log_session_end()
    print(f"Logged test events to {logger.log_file}")
