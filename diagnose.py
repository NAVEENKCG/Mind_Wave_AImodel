"""
ORBIT AI — Pipeline Diagnostic v4 (Event-Aligned)
===================================================
Tests the model using event-aligned epochs from R04 (motor imagery)
and random chunks from R01 (rest baseline).
"""
import pickle
import numpy as np
import mne
import urllib.request
import tempfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR   = Path(__file__).resolve().parent / "data"

MODEL_PATH = MODELS_DIR / "moabb_csp_lda.pkl"
STATS_PATH = DATA_DIR   / "moabb_stats.pkl"

print("=" * 60)
print("  ORBIT AI - Pipeline Diagnostic v4 (Event-Aligned)")
print("=" * 60)

with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)
with open(STATS_PATH, 'rb') as f:
    stats = pickle.load(f)

step_names = [name for name, _ in pipeline.steps]
uses_ts = 'ts' in step_names
ch_mean = stats['ch_mean']
ch_std  = stats['ch_std']
expected_ch = stats['n_channels']
expected_times = stats['n_times']

print(f"[MODEL] {' -> '.join(step_names)}")
print(f"[STATS] {expected_ch} channels, {expected_times} samples, CV={stats['cv_mean']*100:.1f}%")

def classify_epoch(epoch):
    """Classify a single epoch using the pipeline."""
    # Channel + time alignment
    ep = epoch.copy()
    if ep.shape[0] > expected_ch:
        ep = ep[:expected_ch, :]
    if ep.shape[1] > expected_times:
        ep = ep[:, :expected_times]
    elif ep.shape[1] < expected_times:
        repeats = int(np.ceil(expected_times / ep.shape[1]))
        ep = np.tile(ep, (1, repeats))[:, :expected_times]
    
    if uses_ts:
        for c in range(ep.shape[0]):
            ep[c] = (ep[c] - ch_mean[c]) / ch_std[c]
    
    X = ep[np.newaxis, :, :]
    pred = pipeline.predict(X)[0]
    return "FORWARD" if pred == 1 else "IDLE"


# ── Test 1: Motor Imagery with event alignment ──────────────────
print(f"\n{'='*60}")
print("  MOTOR_IMAGERY R04 (Event-Aligned, expect FORWARD)")
print(f"{'='*60}")

url = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
    temp_path = tmp.name
    with urllib.request.urlopen(req, timeout=60) as resp:
        tmp.write(resp.read())

raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
raw.filter(1, 45, verbose=False)
sfreq = raw.info['sfreq']

# Get event annotations (T0=rest, T1=left_fist, T2=right_fist)
events, event_id = mne.events_from_annotations(raw, verbose=False)
print(f"   Event IDs: {event_id}")
print(f"   Total events: {len(events)}")

# Extract motor imagery events (T1 and T2)
motor_event_ids = []
for name, eid in event_id.items():
    if 'T1' in name or 'T2' in name:
        motor_event_ids.append(eid)

print(f"   Motor event IDs: {motor_event_ids}")

chunk_size = int(sfreq * 1.0)
data = raw.get_data()
fwd_count = 0
idle_count = 0
total = 0

for ev in events:
    if ev[2] not in motor_event_ids:
        continue
    start_sample = ev[0]
    if start_sample + chunk_size > data.shape[1]:
        continue
    epoch = data[:, start_sample:start_sample + chunk_size]
    result = classify_epoch(epoch)
    if result == "FORWARD":
        fwd_count += 1
    else:
        idle_count += 1
    total += 1
    if total <= 10:
        print(f"   Event at sample {start_sample}: -> {result}")

print(f"\n   SUMMARY: FORWARD={fwd_count}/{total}  IDLE={idle_count}/{total}")
raw.close()
Path(temp_path).unlink(missing_ok=True)


# ── Test 2: Resting State R01 (random chunks, expect IDLE) ──────
print(f"\n{'='*60}")
print("  RESTING_STATE R01 (Random Chunks, expect IDLE)")
print(f"{'='*60}")

url = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
    temp_path = tmp.name
    with urllib.request.urlopen(req, timeout=60) as resp:
        tmp.write(resp.read())

raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
raw.filter(1, 45, verbose=False)
data = raw.get_data()
sfreq = raw.info['sfreq']
chunk_size = int(sfreq * 1.0)
n_chunks = min(15, data.shape[1] // chunk_size)

fwd_count = 0
idle_count = 0

for i in range(n_chunks):
    start = i * chunk_size
    epoch = data[:, start:start + chunk_size]
    result = classify_epoch(epoch)
    if result == "FORWARD":
        fwd_count += 1
    else:
        idle_count += 1
    print(f"   Chunk {i+1:2d}: -> {result}")

print(f"\n   SUMMARY: FORWARD={fwd_count}/{n_chunks}  IDLE={idle_count}/{n_chunks}")
raw.close()
Path(temp_path).unlink(missing_ok=True)

print(f"\n{'='*60}")
print("  Diagnostic Complete")
print(f"{'='*60}")
