"""
db6_explore.py - first look at one NinaPro DB6 subject. No modelling yet.

Goal of this script: see the data and answer the open questions BEFORE writing
the loader/baseline:
  1. What is inside each zip (the "a" and "b" files)?
  2. What variables / shapes does a .mat file contain?
  3. Which EMG columns are real (DB6 has 16 columns, 14 electrodes)?
  4. How many windows' worth of rest vs grasp (class balance)?
  5. Which repetition used which object? (decides the train/test split)
  6. Is 50 Hz mains visible in the spectrum? (decides if a notch is needed)

Install:   pip install numpy scipy matplotlib        (h5py only if a file is v7.3)

Usage (run from the folder holding the zips):
  python db6_explore.py --zip DB6_s1_a.zip DB6_s1_b.zip --out data/s1
  python db6_explore.py --dir data/s1                       # already unzipped
  python db6_explore.py --dir data/s1 --file S1_D1_T1.mat --seconds 20
  python db6_explore.py --dir data/s1 --all                 # one summary line per file

Figures are saved to <out>/figures/. Add --show to open them on screen.
"""
import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

FS_DEFAULT = 2000          # DB6 sampling rate (Hz), per the Ninapro DB6 page
NAME_RE = re.compile(r"S(\d+)_D(\d+)_T(\d+)", re.IGNORECASE)


# ----------------------------------------------------------------- loading
def unzip_all(zips, out):
    out.mkdir(parents=True, exist_ok=True)
    for z in zips:
        z = Path(z)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            print(f"\n[{z.name}] {len(names)} entries")
            for n in names[:15]:
                print("   ", n)
            if len(names) > 15:
                print(f"    ... (+{len(names) - 15} more)")
            zf.extractall(out / z.stem)   # one subfolder per zip, so a/b stay separate


def load_mat(path):
    """Return dict name -> ndarray. Handles classic .mat and v7.3 (HDF5)."""
    try:
        raw = sio.loadmat(str(path))
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise SystemExit("This .mat is v7.3 - run: pip install h5py")
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    out[k] = np.array(f[k]).T   # HDF5 stores transposed
        return out


def col(d, *names):
    """First variable that exists, as a flat 1-D array (or None)."""
    for n in names:
        if n in d:
            return np.asarray(d[n]).reshape(-1)
    return None


# --------------------------------------------------------------- reporting
def list_mat_files(root):
    files = sorted(Path(root).rglob("*.mat"))
    print(f"\nFound {len(files)} .mat files under {root}")
    for f in files:
        m = NAME_RE.search(f.name)
        tag = f"subj={m.group(1)} day={m.group(2)} time={m.group(3)}" if m else "name not in S#_D#_T# form"
        print(f"  {f.relative_to(root)}   {f.stat().st_size / 1e6:6.1f} MB   {tag}")
    return files


def describe(d):
    print("\nVariables:")
    for k, v in d.items():
        extra = ""
        if v.size and np.issubdtype(v.dtype, np.number) and v.size > 1:
            extra = f"  min={np.nanmin(v):.4g} max={np.nanmax(v):.4g}"
        print(f"  {k:18s} shape={str(v.shape):14s} dtype={v.dtype}{extra}")


def emg_channels(emg, tol=1e-9, quiet=False):
    """Non-constant columns only. This finds EMPTY columns, not dead/noisy electrodes."""
    std = emg.std(axis=0)
    used = [i for i, s in enumerate(std) if s > tol]
    empty = [i for i in range(emg.shape[1]) if i not in used]
    if not quiet:
        print(f"\nEMG matrix: {emg.shape[0]} samples x {emg.shape[1]} columns")
        print(f"  non-constant columns (0-based): {used}  -> {len(used)} channels")
        print(f"  constant/empty columns: {empty}")
    return used


def channel_health(emg, used, quiet=False):
    """Heuristic flags for flat, very noisy or clipped channels. Thresholds are rough
    rules of thumb: treat a flag as 'look at this channel in the raw plot', not a verdict."""
    rms = np.sqrt((emg[:, used] ** 2).mean(axis=0))
    med = float(np.median(rms))
    flagged = {}
    for i, ch in enumerate(used):
        x = emg[:, ch]
        notes = []
        if rms[i] < 0.1 * med:
            notes.append("very low RMS (flat?)")
        if rms[i] > 5 * med:
            notes.append("very high RMS (noisy?)")
        top = float(np.abs(x).max())
        if top > 0 and (np.abs(x) >= 0.999 * top).mean() > 0.001:
            notes.append("many samples at the max (clipped?)")
        if notes:
            flagged[ch] = notes
    if not quiet:
        print("\nChannel health (RMS over the whole file; median RMS = %.4g):" % med)
        for i, ch in enumerate(used):
            tag = "; ".join(flagged.get(ch, [])) or "ok"
            print(f"  ch{ch:2d}  rms={rms[i]:.4g}  {tag}")
        if not flagged:
            print("  no flags - still check the raw plot by eye.")
    return flagged


def mains_ratio(emg, used, fs):
    """50 Hz bin vs its neighbours, computed PER CHANNEL, returned as (median, max) over channels.
    Averaging the PSD across channels would let one noisy channel hide mains in the others."""
    from scipy.signal import welch
    f, pxx = welch(emg[:, used], fs=fs, nperseg=2048, axis=0)
    band = (f >= 49) & (f <= 51)
    side = ((f >= 30) & (f <= 45)) | ((f >= 55) & (f <= 70))
    ratios = pxx[band].max(axis=0) / np.maximum(np.median(pxx[side], axis=0), 1e-30)
    return float(np.median(ratios)), float(ratios.max())


def all_summary(files, fs):
    """One line per .mat file, so Day 1 vs Day 5 etc. can be compared in a single run."""
    print("\nSummary of every file (loads each one; can take a while):")
    print("  file | day time | seconds | rest% | flagged channels | 50Hz peak x (median/max over unflagged channels) | objects by repetition (first grasp)")
    for f in files:
        d = load_mat(f)
        if "emg" not in d:
            print(f"  {f.name}: no 'emg' variable")
            continue
        emg = d["emg"]
        used = emg_channels(emg, quiet=True)
        flagged = channel_health(emg, used, quiet=True)
        lab = col(d, "restimulus", "stimulus")
        rep = col(d, "rerepetition", "repetition")
        obj = col(d, "reobject", "object")
        m = NAME_RE.search(f.name)
        dt = f"D{m.group(2)} T{m.group(3)}" if m else "?"
        rest = f"{100 * (lab == 0).mean():.0f}" if lab is not None else "?"
        sig = "?"
        if lab is not None and rep is not None and obj is not None:
            g = next((v for v in np.unique(lab) if v != 0), None)
            if g is not None:
                msk = lab == g
                parts = []
                for r in np.unique(rep[msk]):
                    if r > 0:
                        o = np.unique(obj[msk & (rep == r)])
                        o = o[o > 0]
                        parts.append(str(int(o[0])) if len(o) else "?")
                sig = ",".join(parts)
        mr = mains_ratio(emg, [c for c in used if c not in flagged] or used, fs)   # skip flagged channels
        print(f"  {f.name} | {dt} | {emg.shape[0] / fs:.0f} | {rest} | "
              f"{sorted(flagged) or 'none'} | {mr[0]:.1f} / {mr[1]:.1f} | {sig}")


def label_report(d):
    lab = col(d, "restimulus", "stimulus")
    rep = col(d, "rerepetition", "repetition")
    obj = col(d, "reobject", "object")
    name_lab = "restimulus" if "restimulus" in d else "stimulus"
    if lab is None:
        print("\nNo stimulus/restimulus variable found - check the variable list above.")
        return
    print(f"\nLabel variable used: {name_lab}")
    vals, cnt = np.unique(lab, return_counts=True)
    for v, c in zip(vals, cnt):
        print(f"  label {int(v):2d}: {c:8d} samples  ({100 * c / lab.size:5.1f}%)")
    rest_frac = float((lab == 0).mean())
    print(f"  rest (label 0) = {100 * rest_frac:.1f}% of samples -> use balanced accuracy,")
    print("  and report false activations at rest separately.")

    if rep is not None and obj is not None:
        print("\nRepetition -> object map per grasp (this decides the split):")
        for g in vals:
            if g == 0:
                continue
            m = lab == g
            reps = np.unique(rep[m])
            reps = reps[reps > 0]
            pairs = []
            for r in reps:
                objs = np.unique(obj[m & (rep == r)])
                objs = objs[objs > 0]
                pairs.append(f"rep{int(r)}:obj{','.join(str(int(o)) for o in objs)}")
            print(f"  grasp {int(g)}: " + "  ".join(pairs))
        print("  If object follows the repetition halves (e.g. reps 1-6 one object, 7-12 another),")
        print("  'train 1-8 / test 9-12' puts a different object only in test. Choose a split that")
        print("  keeps both objects in train AND test, then confirm with this table.")

    for nm in ("daytesting", "time"):
        if nm in d:
            print(f"  {nm}: unique values {np.unique(d[nm])}")


# ----------------------------------------------------------------- figures
def make_figures(emg, used, d, fs, seconds, outdir, name, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    outdir.mkdir(parents=True, exist_ok=True)
    n = min(int(seconds * fs), emg.shape[0])
    t = np.arange(n) / fs
    lab = col(d, "restimulus", "stimulus")

    fig, axes = plt.subplots(len(used) + 1, 1, figsize=(12, 1.0 * (len(used) + 1) + 1), sharex=True)
    for ax, ch in zip(axes[:-1], used):
        ax.plot(t, emg[:n, ch], lw=0.4)
        ax.set_ylabel(f"ch{ch}", rotation=0, labelpad=18, fontsize=7)
        ax.tick_params(labelsize=6)
    if lab is not None:
        axes[-1].plot(t, lab[:n], color="k", lw=0.8)
        axes[-1].set_ylabel("label", rotation=0, labelpad=18, fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{name}: raw EMG, first {seconds:.0f} s (label 0 = rest)")
    fig.tight_layout()
    p1 = outdir / f"{name}_raw.png"
    fig.savefig(p1, dpi=130)

    # PSD (median across channels, so one noisy channel cannot hide a peak)
    f, pxx = welch(emg[:, used], fs=fs, nperseg=2048, axis=0)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.semilogy(f, np.median(pxx, axis=1))
    for hz in (50, 100, 150):
        ax2.axvline(hz, color="r", ls=":", lw=0.8)
    ax2.set_xlim(0, min(500, fs / 2))
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("median PSD across channels")
    ax2.set_title(f"{name}: spectrum (red dotted = 50/100/150 Hz). Notch only if a peak is visible.")
    fig2.tight_layout()
    p2 = outdir / f"{name}_psd.png"
    fig2.savefig(p2, dpi=130)
    print(f"\nSaved figures:\n  {p1}\n  {p2}")
    if show:
        plt.show()


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", nargs="+", help="one or more DB6 zip files to extract")
    ap.add_argument("--out", default="data/db6_subject", help="extraction / output folder")
    ap.add_argument("--dir", help="folder that already holds extracted .mat files")
    ap.add_argument("--file", help="name of the .mat file to inspect (default: first found)")
    ap.add_argument("--fs", type=float, default=FS_DEFAULT, help="sampling rate in Hz (DB6: 2000)")
    ap.add_argument("--seconds", type=float, default=12.0, help="seconds of raw EMG to plot")
    ap.add_argument("--show", action="store_true", help="open figures on screen")
    ap.add_argument("--all", action="store_true", help="print one summary line per .mat file and exit")
    a = ap.parse_args()

    out = Path(a.out)
    if a.zip:
        unzip_all(a.zip, out)
        root = out
    elif a.dir:
        root = Path(a.dir)
    else:
        ap.error("give --zip ... or --dir ...")

    files = list_mat_files(root)
    if not files:
        raise SystemExit("No .mat files found. Check the zip contents printed above.")
    if a.all:
        all_summary(files, a.fs)
        return
    if a.file:
        matches = [f for f in files if f.name == a.file]
        if not matches:
            raise SystemExit(f"--file {a.file!r} not found. Available: " + ", ".join(f.name for f in files))
        target = matches[0]
    else:
        target = files[0]
        print("\nNo --file given: using the first file found. Pass --file to choose (e.g. a Day 5 file).")
    print(f"\nInspecting: {target}")

    d = load_mat(target)
    describe(d)
    if "emg" not in d:
        raise SystemExit("No 'emg' variable - check the variable list above.")
    used = emg_channels(d["emg"])
    channel_health(d["emg"], used)
    print(f"  duration at {a.fs:.0f} Hz: {d['emg'].shape[0] / a.fs:.1f} s")
    label_report(d)
    make_figures(d["emg"], used, d, a.fs, a.seconds, out / "figures", target.stem, a.show)


if __name__ == "__main__":
    main()
