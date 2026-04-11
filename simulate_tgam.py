import socket
import json
import time
import re
import numpy as np
import sys
import urllib.request
import mne
import tempfile
from pathlib import Path
from config import *
from scipy.signal import welch

# Standard EEG frequency bands
BANDS = [
    (0.5, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45), (45, 50)
]

def get_db_power(data_1d, sfreq, fmin, fmax):
    """Calculate power spectral density in Decibels for a frequency band."""
    nperseg = min(len(data_1d), int(sfreq * 1.0))
    if nperseg < 8: return -120.0
    f, psd = welch(data_1d, sfreq, nperseg=nperseg)
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx): return -120.0
    return 10 * np.log10(np.mean(psd[idx]) + 1e-12)


class EEGStreamSimulator:
    """
    Downloads any EDF brainwave file from a URL, extracts spectral features,
    and streams them over a local socket for real-time BCI prediction.
    
    Accepts EDF files from any source (PhysioNet, OpenNeuro, clinical labs, etc).
    """
    def __init__(self):
        self.samples = []
        self.sample_counter = 0
        self.feature_names = [
            "delta", "theta", "lowAlpha", "highAlpha", "lowBeta",
            "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"
        ]

    def setup(self):
        """Prompt user for an EDF URL, download and extract features."""
        print("\n" + "=" * 55)
        print("  🧠 ORBIT AI — EEG SIGNAL STREAMER")
        print("=" * 55)
        print("  Paste any URL pointing to an .edf brainwave file.")
        print("  Supported: PhysioNet, OpenNeuro, clinical labs, etc.")
        print("=" * 55)
        
        url = input("\n🔗 EDF URL: ").strip()
        
        # --- URL AUTO-CORRECTION ---
        # Fix PhysioNet /content/ vs /files/ links
        if "/content/" in url:
            url = url.replace("/content/", "/files/")
            print("🔄 Corrected PhysioNet URL to direct download format.")
        
        # Strip accidental .event, .html, etc. suffixes
        sanitized = re.sub(r'\.edf\..*', '.edf', url, flags=re.IGNORECASE)
        if sanitized != url:
            url = sanitized
            print(f"🧹 Cleaned suffix. Using: {url}")
        
        if not url.lower().endswith(".edf"):
            print("❌ Invalid URL. Must point to an .edf file.")
            sys.exit(1)
        
        # --- DOWNLOAD & EXTRACT ---
        print(f"📡 Downloading from {url}...")
        temp_path = None
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                temp_path = tmp.name
                with urllib.request.urlopen(req) as response:
                    tmp.write(response.read())
            
            raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)
            
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            n_ch, n_samples = data.shape
            chunk_size = int(sfreq * 1.0)
            
            print(f"🧪 Analyzing {n_ch} channels × {n_samples/sfreq:.1f}s of brainwaves...")
            
            for start in range(0, n_samples - chunk_size, chunk_size // 2):
                chunk = data[:, start : start + chunk_size]
                
                band_powers = [
                    np.mean([get_db_power(chunk[c], sfreq, fmin, fmax) for c in range(n_ch)])
                    for fmin, fmax in BANDS
                ]
                
                # Attention = Beta dominance over Alpha (concentration biomarker)
                attn = np.mean(band_powers[4:6]) - np.mean(band_powers[2:4])
                # Meditation = Alpha dominance over Theta (relaxation biomarker)
                med = np.mean(band_powers[2:4]) - band_powers[1]
                
                self.samples.append(band_powers + [attn, med, 0.0])
            
            print(f"✅ Extracted {len(self.samples)} samples for prediction!")
            
            # Cleanup (Windows-safe)
            try:
                raw.close(); del raw
                time.sleep(0.1)
                Path(temp_path).unlink()
            except:
                pass
        
        except Exception as e:
            print(f"❌ Error: {e}")
            if temp_path and Path(temp_path).exists():
                try: Path(temp_path).unlink()
                except: pass
            sys.exit(1)

    def get_packet(self):
        """Return the next feature packet as a dictionary."""
        row = self.samples[self.sample_counter % len(self.samples)]
        self.sample_counter += 1
        return {name: float(val) for name, val in zip(self.feature_names, row)}

    def run(self):
        """Start the streaming server."""
        self.setup()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 9999))
            s.listen(1)
            print(f"\n📡 Streaming on 127.0.0.1:9999")
            print("⏳ Waiting for Dashboard to connect...\n")
            
            while True:
                s.settimeout(None)
                conn, addr = s.accept()
                print(f"✅ Dashboard connected from {addr}")
                self.sample_counter = 0
                
                with conn:
                    while True:
                        try:
                            packet = self.get_packet()
                            data = (json.dumps(packet) + "\n").encode()
                            conn.sendall(data)
                            print(".", end="", flush=True)
                            time.sleep(0.1)
                        except (ConnectionResetError, BrokenPipeError):
                            print("\n❌ Dashboard disconnected.")
                            break
                        except Exception as e:
                            print(f"\n⚠ Stream error: {e}")
                            break


if __name__ == "__main__":
    EEGStreamSimulator().run()
