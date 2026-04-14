"""
ORBIT AI — BioAmp EXG Pill Bridge
==================================
Reads raw EEG/EMG data from an Arduino/BioAmp via Serial,
applies real-time filtering, and streams it to the 
Orbit AI Dashboard.

Wiring:
BioAmp OUT  -> Arduino A0
BioAmp VCC  -> Arduino 5V
BioAmp GND  -> Arduino GND
"""

import serial
import socket
import json
import time
import numpy as np
from scipy.signal import butter, lfilter

# --- CONFIGURATION ---
SERIAL_PORT = "COM3"  # Change this to your Arduino port (e.g., COM3, COM4)
BAUD_RATE   = 115200
SFREQ       = 256.0   # Sampling frequency (matches Arduino delay)
CHANNELS    = 1       # BioAmp is single channel

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

class BioAmpBridge:
    def __init__(self):
        self.raw_buffer = []
        self.chunk_size = int(SFREQ * 1.0) # 1 second chunks
        
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"✅ Connected to BioAmp on {SERIAL_PORT}")
        except Exception as e:
            print(f"❌ Could not open Serial Port {SERIAL_PORT}: {e}")
            print("   Check your Arduino connection and COM port number.")
            exit(1)

    def get_packet(self):
        """Reads from Serial until a full 1s epoch is collected."""
        while len(self.raw_buffer) < self.chunk_size:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line.isdigit():
                self.raw_buffer.append(float(line))
        
        # 1. Take the chunk and clear buffer
        chunk = np.array(self.raw_buffer[:self.chunk_size])
        self.raw_buffer = self.raw_buffer[self.chunk_size:]
        
        # 2. Apply filtering (Remove 50/60Hz hum and DC offset)
        filtered = bandpass_filter(chunk, 1.0, 45.0, SFREQ)
        
        # 3. Format as MOABB-style multi-channel packet (duplicating for 64-channel compatibility)
        # Our dashboard expects 64 channels. We mirror the BioAmp signal across them.
        epoch_64ch = [filtered.tolist()] * 64 

        return {
            "attention": 0.0, # Placeholder
            "meditation": 0.0,
            "_raw_epoch": epoch_64ch,
            "_sfreq": SFREQ,
            "_n_channels": 64
        }

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 9999))
            s.listen(1)
            print(f"\n📡 BioAmp Bridge listening for Dashboard...")

            while True:
                conn, addr = s.accept()
                print(f"✅ Dashboard linked to BioAmp hardware!")
                with conn:
                    while True:
                        try:
                            packet = self.get_packet()
                            data = (json.dumps(packet) + "\n").encode()
                            conn.sendall(data)
                        except Exception as e:
                            print(f"⚠ Connection lost: {e}")
                            break

if __name__ == "__main__":
    BioAmpBridge().run()
