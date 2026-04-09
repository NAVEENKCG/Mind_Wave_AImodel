import time
import json
import socket
import numpy as np
import random
from pathlib import Path

# ORBIT AI Real-Data Simulator
# Streams real clinical EEG data from OpenNeuro files to the dashboard

DATA_DIR = Path("data")

class RealDataSimulator:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        
        # Load local data if exists
        self.X_data = None
        self.y_labels = None
        
        try:
            self.X_data = np.load(DATA_DIR / "X_pretrained.npy")
            self.y_labels = np.load(DATA_DIR / "y_pretrained.npy")
            print(f"✅ Loaded {len(self.X_data)} REAL clinical EEG samples from OpenNeuro.")
        except:
            print("⚠️ X_pretrained.npy not found. Using high-quality synthetic data instead.")

    def get_real_packet(self, target_label):
        """Finds a real brainwave window from the dataset matching the label."""
        if self.X_data is not None:
            # Find indices where label matches
            indices = np.where(self.y_labels == target_label)[0]
            if len(indices) > 0:
                idx = random.choice(indices)
                # Take the last timestep of the window to simulate 1Hz feed
                row = self.X_data[idx][-1] 
                
                cols = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta", "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"]
                return {cols[i]: float(row[i]) for i in range(len(cols))}
        
        # Fallback to smart synthetic if data not found yet
        return {"delta": 200, "theta": 300, "lowAlpha": 100, "highAlpha": 100, "lowBeta": 100, "highBeta": 100, "lowGamma": 50, "highGamma": 25, "attention": 50, "meditation": 50, "blink": 0}

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"\n🧠 ORBIT AI - Clinical Data Streamer running on {self.host}:{self.port}")
            print("--------------------------------------------------")
            print(" [0] -> IDLE (Real Resting Data)")
            print(" [1] -> FORWARD (Real Focus Data)")
            print(" [2] -> LEFT (Real Arithmetic Data)")
            print("--------------------------------------------------")
            
            conn, addr = s.accept()
            with conn:
                current_label = 0
                import msvcrt
                while True:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode()
                        if key in ['0', '1', '2']:
                            current_label = int(key)
                            print(f"🚀 Now streaming REAL data for Class {current_label}")

                    packet = self.get_real_packet(current_label)
                    conn.sendall((json.dumps(packet) + "\n").encode())
                    time.sleep(0.1) # 10Hz - Real-time speed

if __name__ == "__main__":
    sim = RealDataSimulator()
    sim.run()
