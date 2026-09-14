# Model 1 — Universal Brain (Universal BCI Model)

## Overview
**Model 1** is the foundational universal classifier of the ORBIT AI dual-model architecture. It is trained on clinical motor imagery data (PhysioNet EEGMMIDB) across 21 sensorimotor strip channels (FC5–FC6, C5–C6, CP5–CP6) using a Riemannian geometry pipeline (Filter Bank CSP + Tangent Space projection + Linear Discriminant Analysis with Platt calibration).

## Immutability Guarantee
- **Read-Only**: Model 1 is frozen and never updated at runtime.
- Long-term cognitive, impedance, or session variations are handled by **Model 2** (`adapt_drift.py` → `models/drift_profile.json`), preserving baseline integrity.

## Dedicated Module
The standalone execution engine for Model 1 is:
- **`universal_model.py`**: Encapsulates model loading, channel alignment, spatial covariance checks, Platt-calibrated multi-class inference, and latency benchmarks.

## Specifications
- **Pipeline**: Filter Bank Common Spatial Patterns + Riemannian Tangent Space + LDA (`FBCSP_LDA`)
- **Sampling Rate**: 160 Hz
- **Window Length**: 1.0 second (160 samples)
- **Channels**: 21 motor-cortex channels
- **Classes**: 
  - `0`: IDLE (Rest baseline)
  - `1`: LEFT (Left hand imagery)
  - `2`: RIGHT (Right hand imagery)
  - `3`: FORWARD / LIFT (Feet motor imagery)
- **Validation Accuracy**: 83.71% (5-fold stratified cross-validation, leakage-free)
- **Primary Binary File**: `models/moabb_csp_lda.pkl` (gitignored due to size)
- **Metadata Stats**: `data/moabb_stats.pkl`

## Usage
```bash
# Inspect Model 1 specs and status
python universal_model.py

# Run synthetic inference benchmark
python universal_model.py --test

# Verify immutability & file integrity
python universal_model.py --verify
```
