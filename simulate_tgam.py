import socket
import json
import time
import numpy as np
import threading
import sys
import urllib.request
import mne
import tempfile
from pathlib import Path
from config import *
from scipy.signal import welch

# Import the same bands used in training for consistency
BANDS = [
    (0.5, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45), (45, 50)
]

def get_db_power(data_1d, sfreq, fmin, fmax):
    nperseg = min(len(data_1d), int(sfreq * 1.0))
    if nperseg < 8: return -120.0
    f, psd = welch(data_1d, sfreq, nperseg=nperseg)
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx): return -120.0
    return 10 * np.log10(np.mean(psd[idx]) + 1e-12)

class RealDataSimulator:
    def __init__(self):
        self.mode = None
        self.url_data = [] # For Mode 2
        self.local_data = None
        self.class_indices = {}
        self.current_label = 0
        self.sample_counter = 0
        self.feature_names = ["delta","theta","lowAlpha","highAlpha","lowBeta",
                             "highBeta","lowGamma","highGamma","attention","meditation","blink"]

    def select_mode(self):
        print("\n" + "="*50)
        print(" 🧠 ORBIT AI - MULTI-MODE SIMULATOR")
        print("="*50)
        print(" [1] LOCAL CLINICAL DATA (Manual 0/1)")
        print(" [2] REMOTE URL (Paste PhysioNet link)")
        print("="*50)
        
        choice = input("Select Mode (1/2): ").strip()
        if choice == '2':
            self.mode = 'URL'
            self.setup_url_mode()
        else:
            self.mode = 'LOCAL'
            self.setup_local_mode()

    def setup_local_mode(self):
        try:
            self.local_data = np.load(DATA_DIR / "X_pretrained.npy")
            labels = np.load(DATA_DIR / "y_pretrained.npy")
            for cls in [0, 1]:
                self.class_indices[cls] = np.where(labels == cls)[0]
            print(f"✅ Loaded {len(self.local_data)} clinical samples.")
        except:
            print("❌ No local features found. Run quick_process.py first.")
            sys.exit(1)

    def setup_url_mode(self):
        url = input("\n🔗 Paste PhysioNet EDF URL: ").strip()
        
        # Auto-fix PhysioNet view URL vs download URL
        if "/content/" in url.lower():
            url = url.replace("/content/", "/files/")
            print(f"🔄 Corrected PhysioNet URL to direct download: {url}")

        if not url.lower().endswith(".edf"):
            print("⚠️ Warning: URL should end in .edf (not .event or .html).")
            if ".edf" in url.lower():
                url = url.split(".edf")[0] + ".edf"
                print(f"🔄 Auto-corrected extension to: {url}")

        print(f"📡 Downloading signal from {url}...")
        temp_path = None
        try:
            # Step 1: Download to a temporary file
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                temp_path = tmp.name
                with urllib.request.urlopen(req) as response:
                    tmp.write(response.read())
            
            # Step 2: Read with MNE (After closing the file handle)
            raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)
            
            # Step 3: Extract features
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            n_ch, n_samples = data.shape
            chunk_size = int(sfreq * 1.0)
            
            print(f"🧪 Analyzing {n_ch} channels of brainwaves...")
            for start in range(0, n_samples - chunk_size, chunk_size // 2):
                chunk = data[:, start : start + chunk_size]
                
                # Get raw dB powers
                band_powers = [np.mean([get_db_power(chunk[c], sfreq, fmin, fmax) for c in range(n_ch)]) 
                               for fmin, fmax in BANDS]
                
                attn = np.mean(band_powers[4:6]) - np.mean(band_powers[2:4])
                med  = np.mean(band_powers[2:4]) - band_powers[1]
                
                self.url_data.append(band_powers + [attn, med, 0.0])
            
            print(f"✅ Successfully extracted {len(self.url_data)} samples for prediction!")
            
            # Step 4: Cleanup (optional, Windows-safe)
            try:
                raw.close()
                del raw
                time.sleep(0.1) # Small delay to release file
                Path(temp_path).unlink()
            except:
                pass # If Windows still locks it, just leave it for the OS to clean later

        except Exception as e:
            print(f"❌ Error downloading/processing URL: {e}")
            if temp_path and Path(temp_path).exists():
                try: Path(temp_path).unlink() 
                except: pass
            sys.exit(1)

    def get_packet(self):
        if self.mode == 'LOCAL':
            indices = self.class_indices[self.current_label]
            idx = indices[self.sample_counter % len(indices)]
            self.sample_counter += 1
            row = self.local_data[idx][len(self.local_data[idx]) // 2]
        else:
            row = self.url_data[self.sample_counter % len(self.url_data)]
            self.sample_counter += 1
            
        return {name: float(val) for name, val in zip(self.feature_names, row)}

    def keyboard_listener(self):
        if self.mode != 'LOCAL': return
        while True:
            key = input().strip()
            if key in ['0', '1']:
                self.current_label = int(key)
                print(f"🔄 Switched to Class {key}")

    def run(self):
        self.select_mode()
        kb_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        kb_thread.start()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 9999))
            s.listen(1)
            print(f"\n📡 Simulator READY on 127.0.0.1:9999")
            print("⏳ WAITING for Dashboard to connect...")
            
            while True:
                s.settimeout(None)
                conn, addr = s.accept()
                print(f"\n✅ Dashboard joined from {addr}")
                self.sample_counter = 0
                
                with conn:
                    while True:
                        try:
                            packet = self.get_packet()
                            # Convert to binary and send
                            data = (json.dumps(packet) + "\n").encode()
                            conn.sendall(data)
                            
                            # Visual feedback
                            print(".", end="", flush=True)
                            time.sleep(0.1)
                        except (ConnectionResetError, BrokenPipeError):
                            print("\n❌ Dashboard disconnected.")
                            break
                        except Exception as e:
                            print(f"\n⚠ Stream error: {e}")
                            break

if __name__ == "__main__":
    sim = RealDataSimulator()
    sim.run()
