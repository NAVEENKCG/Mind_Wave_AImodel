import serial
import time
import json
import socket
import numpy as np
from scipy.signal import butter, lfilter

# ── CONFIGURATION ──────────────────────────────────────────────────────────
SERIAL_PORT = "COM3"  # Change this to your Arduino port
BAUD_RATE = 115200
SAMPLE_RATE = 100     # AD8232 is usually sampled at 100Hz - 200Hz
SOCKET_PORT = 9999
CHANNELS = 64         # Our AI model expects 64 logic channels (we duplicate the single channel)

# ── SIGNAL PROCESSING ──────────────────────────────────────────────────────
# Since AD8232 is noisy, we use a tighter bandpass filter (3Hz - 30Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=3.0, highcut=30.0, fs=100.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# ── MAIN BRIDGE ────────────────────────────────────────────────────────────
def main():
    print(f"🚀 [AD8232 BRIDGE] Initializing budget sensor on {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', SOCKET_PORT))
        print("✅ Connected to Arduino and Dashboard!")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    buffer = []
    
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line and line.isdigit():
                val = int(line)
                
                # Normalize 10-bit analog (0-1023) to microvolts range roughly
                val = (val - 512) * 0.5 
                buffer.append(val)
                
                if len(buffer) >= 10:  # Process in small chunks
                    filtered = bandpass_filter(buffer, fs=SAMPLE_RATE)
                    
                    # Package for the Dashboard
                    # We broadcast the single sensor across all 64 channels the AI expects
                    latest_val = filtered[-1]
                    packet = {
                        "data": [latest_val] * CHANNELS
                    }
                    
                    client_socket.sendall(json.dumps(packet).encode('utf-8') + b'\n')
                    buffer = buffer[-5:] # Keep a small overlap for smooth filtering
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping AD8232 Bridge...")
            break
        except Exception as e:
            print(f"⚠ Signal glitch: {e}")
            break

    ser.close()
    client_socket.close()

if __name__ == "__main__":
    main()
