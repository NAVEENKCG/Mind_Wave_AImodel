import socket
import json
import time
import numpy as np
import threading
import sys
from pathlib import Path
from config import *

class RealDataSimulator:
    def __init__(self):
        self.data = np.load(DATA_DIR / "X_pretrained.npy")
        self.labels = np.load(DATA_DIR / "y_pretrained.npy")
        print(f"✅ Loaded {len(self.data)} REAL clinical EEG samples from OpenNeuro.\n")
        
        # Only use IDLE (0) and FORWARD (1)
        self.class_indices = {}
        for cls in [0, 1]:
            idx = np.where(self.labels == cls)[0]
            if len(idx) > 0:
                self.class_indices[cls] = idx
        
        self.current_label = 0  # Start at IDLE
        self.sample_counter = {0: 0, 1: 0}
        self.feature_names = ["delta","theta","lowAlpha","highAlpha","lowBeta","highBeta","lowGamma","highGamma","attention","meditation","blink"]

    def get_real_packet(self, label):
        """Pick the next real sample for the given class."""
        indices = self.class_indices.get(label, self.class_indices[0])
        idx = indices[self.sample_counter[label] % len(indices)]
        self.sample_counter[label] += 1
        
        # Get one row from the window (middle row for stability)
        window = self.data[idx]
        row = window[len(window) // 2]
        
        packet = {}
        for i, name in enumerate(self.feature_names):
            if i < len(row):
                packet[name] = float(row[i])
            else:
                packet[name] = 0.0
        return packet

    def keyboard_listener(self):
        """Listen for keyboard input to change class."""
        while True:
            try:
                key = input()
                if key.strip() == "0":
                    self.current_label = 0
                    print(f"🔄 Switched to Class 0 → IDLE (Resting)")
                elif key.strip() == "1":
                    self.current_label = 1
                    print(f"🔄 Switched to Class 1 → FORWARD (Focus)")
                else:
                    print(f"   Press 0 (IDLE) or 1 (FORWARD)")
            except EOFError:
                break

    def run(self):
        # Start keyboard listener in background
        kb_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        kb_thread.start()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 9999))
            s.listen(1)
            
            print("🧠 ORBIT AI - Clinical Data Streamer running on 127.0.0.1:9999")
            print("─" * 50)
            print(" [0] → IDLE (Resting Brain)")
            print(" [1] → FORWARD (Focused Brain)")
            print("─" * 50)
            print(f"▶ Currently streaming: Class {self.current_label}")
            print("Type 0 or 1 and press Enter to switch.\n")
            
            while True:
                try:
                    conn, addr = s.accept()
                    print(f"📡 Dashboard connected from {addr}")
                    
                    with conn:
                        prev_label = -1
                        while True:
                            if self.current_label != prev_label:
                                print(f"🚀 Now streaming REAL data for Class {self.current_label}")
                                prev_label = self.current_label
                            
                            packet = self.get_real_packet(self.current_label)
                            conn.sendall((json.dumps(packet) + "\n").encode())
                            time.sleep(0.1)  # 10Hz
                            
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    print("⚠️ Dashboard disconnected. Waiting for reconnection...")
                    continue

if __name__ == "__main__":
    sim = RealDataSimulator()
    sim.run()
