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
        print(f"✅ Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Could not open Serial Port {SERIAL_PORT}: {e}")
        print("   Check your Arduino connection and COM port number.")
        return

    buffer = []
    chunk_size = int(SAMPLE_RATE * 1.0)  # 1 second of data

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('127.0.0.1', SOCKET_PORT))
        server_socket.listen(1)
        print(f"\n📡 AD8232 Bridge listening on 127.0.0.1:{SOCKET_PORT}")
        print("⏳ Waiting for Dashboard to connect...")

        while True:
            conn, addr = server_socket.accept()
            print(f"✅ Dashboard connected from {addr}")
            buffer = []

            with conn:
                while True:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if line and line.isdigit():
                            val = int(line)
                            # Normalize 10-bit analog (0-1023) to microvolts range roughly
                            val = (val - 512) * 0.5
                            buffer.append(val)

                            if len(buffer) >= chunk_size:
                                filtered = bandpass_filter(buffer[:chunk_size], fs=SAMPLE_RATE)

                                # Duplicate single channel across CHANNELS to match model expectation
                                epoch = [filtered.tolist()] * CHANNELS

                                packet = {
                                    "attention": 0.0,
                                    "meditation": 0.0,
                                    "_raw_epoch": epoch,
                                    "_sfreq": float(SAMPLE_RATE),
                                    "_n_channels": CHANNELS
                                }

                                conn.sendall(json.dumps(packet).encode('utf-8') + b'\n')
                                buffer = buffer[chunk_size // 2:]  # 50% overlap

                    except KeyboardInterrupt:
                        print("\n🛑 Stopping AD8232 Bridge...")
                        ser.close()
                        return
                    except (ConnectionResetError, BrokenPipeError):
                        print("\n❌ Dashboard disconnected.")
                        break
                    except Exception as e:
                        print(f"⚠ Signal glitch: {e}")
                        break

if __name__ == "__main__":
    main()
