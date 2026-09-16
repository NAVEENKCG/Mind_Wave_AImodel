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

## 💡 Future Ideas & Model Evolution

The future evolution of Model 1 focuses on eliminating electrode density constraints and enabling edge deployment:

1. **Channel-Agnostic Deep Architecture (`EEGNet-EXG`)**:
   - Transition from 21-channel clinical caps to 1–2 channel bio-potential sensors (e.g. BioAmp EXG Pill) via spatial attention maps, removing the requirement for synthetic zero-padding.
2. **Riemannian Procrustes Analysis (RPA)**:
   - Implement geometric manifold alignment to mathematically align new subject covariance matrices onto the canonical Model 1 Riemannian tangent space without retraining.
3. **Edge Quantization (INT8 & Microcontrollers)**:
   - Export Model 1 weights to ONNX Runtime and TensorFlow Lite Micro with INT8 quantization, lowering memory footprint below 250 KB for direct execution on Raspberry Pi 5 or ESP32-S3.

---
*Reference: [Main Innovation Roadmap](../../README.md#💡-innovation-roadmap--future-ideas) · [System Architecture](../../ARCHITECTURE.md)*

