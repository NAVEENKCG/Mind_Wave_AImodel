# ORBIT AI — File Roles and Responsibilities

This document provides a clear, comprehensive reference mapping each script and file in the ORBIT AI project to its specific role and function in the system architecture.

---

## 🌉 1. Signal Bridges & Simulators (Getting Data In)

- [simulate_tgam.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/simulate_tgam.py): The simulator interface. Streams real clinical EEG recordings from pre-training sets (`X_pretrained.npy`) over local socket `127.0.0.1:9999` to mock the hardware connection for testing and offline demos.
- [bridge_bioamp.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/bridge_bioamp.py): The primary hardware bridge. Reads raw EEG/EXG signals from the BioAmp EXG Pill via a USB serial connection and broadcasts it over local socket.
- [bridge_ad8232.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/bridge_ad8232.py): An alternative budget hardware bridge. Connects to an Arduino reading from an AD8232 sensor, applies custom bandpass filtering, replicates the single input to a 64-channel matrix matching the model architecture, and streams it.

---

## ⚙️ 2. Data Sourcing & Preprocessing (Data Pipeline)

- [fetch_and_process_openneuro.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/fetch_and_process_openneuro.py): Sourcing tool that downloads large clinical EEG datasets (ds002721 & ds003478) from OpenNeuro, extracts power band features from specific electrode channels (like Fp1), and compiles them into pre-training matrices.
- [preprocess.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/preprocess.py): The main offline data processing pipeline. Handles loading raw CSV logs, removing outliers, computing derived ratio features, applying sliding windows, performing robust scaling normalization, and applying data augmentation/undersampling for class balance.
- [quick_process.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/quick_process.py): An accelerated preprocessing script for the ds002721 dataset that uses decibel-scale Welch Power Spectral Density (PSD) analysis to quickly extract normalized training features.

---

## 🧠 3. Brain Engine & Learning (Models & Custom Profiles)

- [model.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/model.py): Defines the neural network architectures in PyTorch, including a Bidirectional LSTM with Self-Attention and a hybrid 1D CNN-LSTM model that captures spatial-temporal patterns in brain waves.
- [train.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/train.py): The core deep learning training script. Loads and filters pre-processed data to binary classes (IDLE vs. FORWARD), balances sample distribution, and trains the CNN-LSTM model with cosine annealing learning rate schedules.
- [train_moabb.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/train_moabb.py): The machine learning model trainer. Uses the MOABB library to fetch medical motor imagery datasets and trains a Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline.
- [fine_tune.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/fine_tune.py): Implements transfer learning by loading a pre-trained base model, freezing feature-extraction layers, and fine-tuning weights at a low learning rate on the user's personalized dataset.
- [calibrate.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/calibrate.py): The profiler script. Conducts a quick user test to measure individual baseline resting levels, active focus states, blink strength, and fatigue thresholds.
- [collect_data.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/collect_data.py): The personal data collector. Interactively guides the user to record specific movement intentions, compiling data directly from the network socket into `personal_raw_data.csv`.

---

## 📊 4. Diagnostics, Logging & Reporting (Verification & Stats)

- [evaluate.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/evaluate.py): Model evaluation interface. Computes detailed classification reports, saves confusion matrices, confidence distributions, and performs permutation feature importance checks.
- [diagnose.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/diagnose.py): Event-aligned offline pipeline tester. Validates models on raw PhysioNet records and reports system dependencies, GPU/CUDA availability, and hardware status.
- [logger_orbit.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/logger_orbit.py): The centralized real-time logging utility. Automatically records timestamps, predicted commands, confidence, and signal quality metrics to daily CSV files.
- [auto_report.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/auto_report.py): Auto-generates post-session reports (text and PDF formats) summarizing command distribution histograms, safety events, and average signal performance.

---

## ⚙️ 5. Real-Time Execution (The Core Engine)

- [predict_realtime.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/predict_realtime.py): The core runtime and user interface dashboard. Listens to the streaming bridge, applies safety gates (noise, signal dropouts, fatigue, and clenches), performs model inference, and renders the virtual wheelchair arena.

---

## 🛠️ 6. Configuration & Documentation

- [config.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/config.py): The configuration panel. Defines variables for COM ports, directories, sampling rates, sliding window sizes, safety limits, and classification target command maps.
- [README.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/README.md): High-level setup guide and user manual for running the ORBIT AI system.
- [visual_workflow.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/visual_workflow.md): Mermaid-based flowcharts tracking data pipelines, offline training, and live inference.
- [ARCHITECTURE.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/ARCHITECTURE.md): Technical deep-dive of the algorithms, feature extractions, models, and multi-tier safety layers.
- [orbit_ai_architecture.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/orbit_ai_architecture.md): The full architectural overview manual, detailing setup workflow diagrams, model layers, and configuration guides.
- [.gitignore](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/.gitignore): Specifies file exclusions to avoid tracking virtual environments, large data files (`.npy`, `.csv`, `.edf`), and model checkpoints (`.pth`, `.pkl`).
- [requirements.txt](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/requirements.txt): Declares required external library dependencies.
