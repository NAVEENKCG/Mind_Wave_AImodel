"""
ORBIT AI — EEG Stream Simulator (MOABB Edition)
================================================
Downloads any EDF brainwave file from a URL or local path,
extracts raw multi-channel epochs, and streams them over a local
socket for real-time CSP+LDA prediction.
"""

import socket
import json
import time
import re
import sys
import numpy as np
import urllib.request
import urllib.error
import mne
import tempfile
from pathlib import Path
from config import *
from scipy.signal import welch

# Standard EEG frequency bands (for legacy attention feature)
BANDS = [
    (0.5, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45), (45, 50)
]

def get_db_power(data_1d, sfreq, fmin, fmax):
    """Calculate power spectral density in dB for a frequency band."""
    nperseg = min(len(data_1d), int(sfreq * 1.0))
    if nperseg < 8: return -120.0
    from scipy.signal import welch as _welch
    f, psd = _welch(data_1d, sfreq, nperseg=nperseg)
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx): return -120.0
    return 10 * np.log10(np.mean(psd[idx]) + 1e-12)


class EEGStreamSimulator:
    """
    Streams EEG data from any .edf file (URL or local path).
    Sends both:
      - Band power features (for BioSensor fallback)
      - Raw channel epoch (for MOABB CSP+LDA classifier)
    """
    def __init__(self):
        self.samples    = []       # List of dicts: {band_powers, raw_epoch, sfreq, n_ch}
        self.sample_counter = 0
        self.sfreq      = None
        self.n_channels = None
        self.feature_names = [
            "delta", "theta", "lowAlpha", "highAlpha", "lowBeta",
            "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"
        ]

    # ── Setup ────────────────────────────────────────────────────────
    def setup(self):
        print("\n" + "=" * 55)
        print("  🧠 ORBIT AI — EEG SIGNAL STREAMER")
        print("=" * 55)
        print("  Paste a URL (PhysioNet, etc.) OR a local file path.")
        print("  Supported formats: .edf")
        print("=" * 55)

        source = input("\n🔗 EDF Source (URL or path): ").strip().strip('"')
        edf_path = self._resolve_source(source)
        self._load_and_extract(edf_path)

    def _resolve_source(self, source):
        """Accept either a URL or a local filesystem path."""
        # ── Local file path ─────────────────────────────────────────
        p = Path(source)
        if p.exists() and p.suffix.lower() == '.edf':
            print(f"📂 Using local file: {p}")
            return str(p)

        # ── URL ─────────────────────────────────────────────────────
        url = source

        # Auto-fix PhysioNet /content/ → /files/
        if "/content/" in url:
            url = url.replace("/content/", "/files/")
            print("🔄 Corrected PhysioNet URL to direct download format.")

        # Strip accidental .event / .html suffixes
        sanitized = re.sub(r'\.edf\..*', '.edf', url, flags=re.IGNORECASE)
        if sanitized != url:
            url = sanitized
            print(f"🧹 Cleaned suffix. Using: {url}")

        if not url.lower().endswith(".edf"):
            print("❌ Invalid source. Must be a .edf URL or local file path.")
            sys.exit(1)

        print(f"📡 Downloading from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                temp_path = tmp.name
                with urllib.request.urlopen(req, timeout=60) as resp:
                    tmp.write(resp.read())
            return temp_path
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            sys.exit(1)

    def _load_and_extract(self, edf_path):
        """Load EDF, extract band power + raw epoch per 1s chunk."""
        is_temp = not Path(edf_path).exists() or edf_path.startswith(tempfile.gettempdir())

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)

            data = raw.get_data()
            self.sfreq = raw.info['sfreq']
            self.n_channels = data.shape[0]
            n_samples = data.shape[1]
            chunk_size = int(self.sfreq * 1.0)

            print(f"🧪 {self.n_channels} channels × {n_samples/self.sfreq:.1f}s @ {self.sfreq:.0f}Hz")
            print("   Extracting features...")

            for start in range(0, n_samples - chunk_size, chunk_size // 2):
                chunk = data[:, start: start + chunk_size]

                # Band powers (for BioSensor fallback)
                band_powers = [
                    np.mean([get_db_power(chunk[c], self.sfreq, f1, f2)
                             for c in range(self.n_channels)])
                    for f1, f2 in BANDS
                ]
                attn = np.mean(band_powers[4:6]) - np.mean(band_powers[2:4])
                med  = np.mean(band_powers[2:4]) - band_powers[1]
                features = band_powers + [attn, med, 0.0]

                self.samples.append({
                    "features"  : features,
                    "raw_epoch" : chunk.tolist(),   # shape: n_ch × chunk_size
                    "sfreq"     : float(self.sfreq),
                    "n_channels": int(self.n_channels),
                })

            raw.close()
            print(f"✅ Extracted {len(self.samples)} samples — ready to stream!")

        except Exception as e:
            print(f"❌ Error loading EDF: {e}")
            sys.exit(1)
        finally:
            # Cleanup temp file (Windows-safe)
            try:
                if is_temp:
                    time.sleep(0.1)
                    Path(edf_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── Streaming ────────────────────────────────────────────────────
    def get_packet(self):
        """Return the next sample as a JSON-serializable dict."""
        sample = self.samples[self.sample_counter % len(self.samples)]
        self.sample_counter += 1

        packet = {name: float(val)
                  for name, val in zip(self.feature_names, sample["features"])}
        packet["_raw_epoch"]  = sample["raw_epoch"]
        packet["_sfreq"]      = sample["sfreq"]
        packet["_n_channels"] = sample["n_channels"]
        return packet

    def run(self):
        """Start the TCP streaming server."""
        self.setup()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 9999))
            s.listen(1)
            print(f"\n📡 Streaming on 127.0.0.1:9999")
            print("⏳ Waiting for Dashboard to connect...\n")

            while True:
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
