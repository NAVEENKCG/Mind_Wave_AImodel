# 🦽 ORBIT AI — Smart Wheelchair BCI Simulator

An interactive assistive mobility dashboard for the **ORBIT AI Brain-Computer Interface**. It visualizes real-time EEG-decoded motor imagery commands navigating a virtual wheelchair across an arena grid, backed by a fast-path EMG emergency brake.

---

## 🌟 Overview

The **Wheelchair BCI Simulator** serves as the primary evaluation environment for assistive mobility before physical motor controller deployment. It bridges the gap between neural intention and robotic navigation:

- **4-Class Motor Imagery Control**: Translates decoded user mental tasks (`FORWARD`, `LEFT`, `RIGHT`, `IDLE`) into directional locomotion.
- **Fail-Safe EMG Emergency Stop**: Instantly halts motion when a jaw clench is detected (< 100ms response time).
- **Live Neural Telemetry**: Real-time visualization of Theta, Alpha, and Beta spectral band powers, signal quality index, and drowsiness/fatigue ratios.
- **WebSocket Streaming**: Connects directly to `predict_websocket.py` (`ws://localhost:8765`) running the Riemannian Model 1 / Model 2 inference engine.

---

## 🕹️ Control & Command Mapping

| Command | Modality | Mental / Physical Task | Wheelchair Action |
| :--- | :---: | :--- | :--- |
| **`FORWARD`** | 🧠 EEG | Imagine moving both feet | Drives wheelchair forward one grid unit |
| **`LEFT`** | 🧠 EEG | Imagine squeezing left hand | Rotates heading 90° counter-clockwise |
| **`RIGHT`** | 🧠 EEG | Imagine squeezing right hand | Rotates heading 90° clockwise |
| **`IDLE`** | 🧠 EEG | Relaxed baseline, open eyes | Neutral / coasting state |
| **`STOP`** | 💪 EMG | **Jaw clench (bite teeth)** | **Emergency Stop** — locks brakes instantly |

---

## 🛠️ Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Framework** | Next.js 15 (App Router), React 19, TypeScript | Server and client component architecture in strict mode |
| **Styling** | Tailwind CSS (CSS Variables) | Dark-first medical grade theme with glassmorphic cards |
| **Animation** | Framer Motion | Smooth coordinate transitions, pulse indicators, and layout springs |
| **Icons** | Lucide React | High-contrast assistive icons |
| **Communication** | Native WebSocket Client | Persistent connection with exponential backoff auto-reconnect |

### Key Components
- **`Arena.tsx`**: High-performance 24×16 grid rendering real-time wheelchair position, orientation vector, and movement trail.
- **`StatusPanel.tsx`**: Connection state, classification confidence meter, signal RMS health, and fatigue status.
- **`BrainPowerPanel.tsx`**: Spectral power gauges for Theta (drowsiness), Alpha (relaxation), and Beta (active motor imagery).

---

## ⚡ Quick Start

### 1. Start the BCI Backend
In your main project root:

```bash
# Terminal 1: Start synthetic EEG stream (or connect real BioAmp hardware)
python simulate_tgam.py

# Terminal 2: Launch the WebSocket inference server
python predict_websocket.py --demo
```

### 2. Launch the Wheelchair Simulator
In this directory (`wheelchair-simulator/`):

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 💡 Future Ideas & Engineering Roadmap

```
  [Decoded Motor Imagery] ────────┐
                                  ├─► [Shared Autonomy VFH+ Engine] ──► [Safe Wheelchair Trajectory]
  [Simulated 2D LiDAR / Sonar] ───┘
```

### 1. 🛡️ Shared-Autonomy Vector Field Navigation
- Implement **Vector Field Histogram (VFH+)** and **Dynamic Window Approach (DWA)**.
- While the user’s brain provides directional guidance, the system automatically detects virtual walls and obstacles, blending control vectors to steer safely through doorways.

### 2. 📡 2D LiDAR & Ultrasonic Sensor Simulation
- Overlay simulated distance scanner rays directly on the arena grid.
- Dynamic obstacle generation (moving pedestrians, sudden barriers) to test reaction times under varying fatigue conditions.

### 3. 🌐 3D Digital Twin (Three.js / WebGL)
- Migrate the 2D grid into an immersive Three.js 3D hospital room and outdoor sidewalk simulation.
- Include wheelchair physics (mass, inertia, rolling friction, slope climbing).

### 4. 🔐 PalmPay Biometric Ignition & Security Lock
- Integrate with [**PalmPay**](../future_ideas.md#1-palmpay--ai-based-palm-vein-biometric-authentication--medical-device-gateway) via Near-Infrared palm-vein scanning.
- Prevent unauthorized movement by requiring the patient’s unique vascular pattern to unlock the wheelchair controller.

### 5. 🔌 Physical Wheelchair Motor Controller Bridge
- Direct serial / CAN-bus hardware interface connecting Next.js / Python outputs to Curtis or Roboteq electric wheelchair motor drivers.

---

## 🔗 Related Documentation
- [ORBIT AI Main Documentation](../README.md)
- [Future Ideas & Comprehensive Roadmap](../future_ideas.md)
- [Prosthetic Arm Simulator](../prosthetic-arm-simulator/README.md)
- [System Architecture](../ARCHITECTURE.md)
