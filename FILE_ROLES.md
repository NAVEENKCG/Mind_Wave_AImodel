# ORBIT AI — File Roles and Responsibilities

This document provides a clear explanation of what each script in the ORBIT AI project does.

## 🌉 1. Signal Bridges (Getting Data In)
- **`simulate_tgam.py`**: The Simulator. It downloads real clinical brainwave data (EDF files) and streams it locally. Use this when you don't have the physical hardware connected.
- **`bridge_bioamp.py`**: The Hardware Bridge. It reads raw analog signals from your BioAmp EXG Pill (via serial port) and streams them locally. Use this when using the physical headset.

## 🧠 2. The Brain Engine (Training & Profiling)
- **`train_moabb.py`**: The AI Trainer. It downloads world-class medical datasets (like PhysioNet) and trains the "Universal Brain" (a machine learning model) to understand general human movement intentions.
- **`calibrate.py`**: The User Profiler. Since every brain is unique, this script runs a 45-second test on *you* to establish your baseline resting state and fatigue thresholds.

## ⚙️ 3. Real-Time Execution (The Core)
- **`predict_realtime.py`**: The Main Dashboard & Safety Controller. This script:
  1. Listens to the incoming brainwaves.
  2. Runs them through 5 Safety Gates (checking for noise, fatigue, jaw clenching).
  3. Uses the AI model to predict your movement intent (`FORWARD` or `IDLE`).
  4. Displays the virtual wheelchair arena.

## 🛠️ 4. Configuration & Docs
- **`config.py`**: Central settings file where thresholds, file paths, and sampling rates are defined.
- **`visual_workflow.md`**: Contains a visual flowchart of how data moves through the entire system.
- **`ARCHITECTURE.md`**: A deep-dive into the technical architecture, safety systems, and machine learning models used.
