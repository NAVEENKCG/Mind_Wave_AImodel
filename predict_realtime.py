import torch
import numpy as np
import serial
import joblib
import time
import logging
import collections
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.console import Console
from rich import box

from config import (
    SERIAL_PORT, BAUD_RATE, WHEELCHAIR_PORT, WHEELCHAIR_BAUD,
    MODELS_DIR, USE_CNN_LSTM, CLASS_NAMES,
    CONFIDENCE_THRESHOLDS, HOLD_REQUIRED, COOLDOWN_SECONDS,
    INFER_WINDOW_SIZE
)
from model import get_model

# Setup logging
logging.basicConfig(level=logging.ERROR) # Only show errors to avoid cluttering Rich UI
console = Console()

class RealTimeInference:
    """Orchestrates live EEG data intake, model prediction, and wheelchair control."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model & Scaler
        self.model = get_model(USE_CNN_LSTM).to(self.device)
        self.model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        
        # Buffers
        self.data_buffer = collections.deque(maxlen=INFER_WINDOW_SIZE)
        self.prediction_buffer = collections.deque(maxlen=3)
        
        # State
        self.last_cmd = "IDLE"
        self.last_cmd_time = 0
        self.hold_count = 0
        self.is_connected = False
        
        # Serial
        try:
            self.ser_tgam = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.ser_wheelchair = serial.Serial(WHEELCHAIR_PORT, WHEELCHAIR_BAUD, timeout=0.1)
            self.is_connected = True
        except Exception as e:
            console.print(f"[red]Serial Connection Error: {e}[/red]")

    def add_derived_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Add derived features to a single timestep [11] -> [18]."""
        # delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, highGamma, attn, med, blink
        epsilon = 1e-6
        bands = raw_features[:8]
        total_power = np.sum(bands)
        
        theta_beta = raw_features[1] / (raw_features[4] + raw_features[5] + epsilon)
        alpha_beta = (raw_features[2] + raw_features[3]) / (raw_features[4] + raw_features[5] + epsilon)
        attn_med_diff = raw_features[8] - raw_features[9]
        beta_ratio = (raw_features[4] + raw_features[5]) / (total_power + epsilon)
        alpha_ratio = (raw_features[2] + raw_features[3]) / (total_power + epsilon)
        theta_ratio = raw_features[1] / (total_power + epsilon)
        
        derived = np.array([
            theta_beta, alpha_beta, attn_med_diff, total_power,
            beta_ratio, alpha_ratio, theta_ratio
        ])
        return np.concatenate([raw_features, derived])

    def parse_tgam(self, line: bytes) -> np.ndarray:
        """Parse raw serial line to feature array."""
        try:
            vals = list(map(int, line.decode('utf-8').strip().split(',')))
            if len(vals) == 11:
                return np.array(vals)
        except:
            pass
        return None

    def get_cmd_char(self, class_name: str) -> str:
        """Map class name to wheelchair serial character."""
        mapping = {"FORWARD": "F\n", "LEFT": "L\n", "RIGHT": "R\n", "STOP": "S\n", "IDLE": "I\n"}
        return mapping.get(class_name, "I\n")

    def run_inference(self):
        """Main loop for live dashboard and control."""
        if not self.is_connected: return

        def generate_dashboard():
            # Build Live Rich Dash
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=5)
            )
            
            # Header
            layout["header"].update(Panel("ORBIT AI — LIVE EEG MONITOR", style="bold magenta", box=box.DOUBLE))
            
            # Sub-components would be updated in the loop
            return layout

        with Live(generate_dashboard(), refresh_per_second=10) as live:
            while True:
                line = self.ser_tgam.readline()
                raw_feats = self.parse_tgam(line)
                
                if raw_feats is not None:
                    # Feature Pipeline
                    feats_18 = self.add_derived_features(raw_feats)
                    self.data_buffer.append(feats_18)
                    
                    # Update Visuals
                    attn, med = raw_feats[8], raw_feats[9]
                    
                    if len(self.data_buffer) == INFER_WINDOW_SIZE:
                        # Prepare for model
                        window = np.array(self.data_buffer)
                        # Scaling (needs 2D [samples, features])
                        window_scaled = self.scaler.transform(window)
                        input_tensor = torch.FloatTensor(window_scaled).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            probs = torch.softmax(outputs, dim=1)[0]
                            conf, pred_idx = torch.max(probs, dim=0)
                        
                        pred_class = CLASS_NAMES[pred_idx.item()]
                        threshold = CONFIDENCE_THRESHOLDS.get(pred_idx.item(), 0.8)
                        
                        # Command Logic
                        if conf.item() >= threshold:
                            self.prediction_buffer.append(pred_class)
                            
                            # Majority Vote & Hold Logic
                            if len(self.prediction_buffer) == 3 and all(p == pred_class for p in self.prediction_buffer):
                                self.hold_count = min(self.hold_count + 1, HOLD_REQUIRED)
                                
                                # Execution under Cooldown
                                if self.hold_count >= HOLD_REQUIRED and (time.time() - self.last_cmd_time) > COOLDOWN_SECONDS.get(pred_idx.item(), 0):
                                    self.last_cmd = pred_class
                                    self.last_cmd_time = time.time()
                                    self.ser_wheelchair.write(self.get_cmd_char(pred_class).encode())
                            else:
                                self.hold_count = 0
                        else:
                            self.prediction_buffer.append("UNCERTAIN")
                            self.hold_count = 0

                    # UI Rendering
                    table = Table(box=None, show_header=False, expand=True)
                    table.add_row(f"Attention:  [{'█'*int(attn/10)}{'░'*(10-int(attn/10))}] {attn}%")
                    table.add_row(f"Meditation: [{'█'*int(med/10)}{'░'*(10-int(med/10))}] {med}%")
                    
                    pred_txt = f"[bold green]{self.last_cmd}[/bold green]" if self.hold_count >= HOLD_REQUIRED else "WAITING..."
                    
                    footer = Table(box=box.SIMPLE, show_header=False, expand=True)
                    footer.add_row(f"Prediction: {pred_txt} | Confidence: {int(conf.item()*100) if 'conf' in locals() else 0}%")
                    footer.add_row(f"Hold State: [{'█'*self.hold_count}{'░'*(HOLD_REQUIRED-self.hold_count)}] {self.hold_count}/{HOLD_REQUIRED}")
                    footer.add_row(f"Last CMD:   {self.last_cmd} sent to {WHEELCHAIR_PORT}")

                    live.update(Panel(table, title="Signals"), refresh=True)
                    # Note: Simplified Rich update for demo purposes
                
                time.sleep(0.01)

if __name__ == "__main__":
    inf = RealTimeInference()
    inf.run_inference()
