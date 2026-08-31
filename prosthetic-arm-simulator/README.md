# prosthetic-arm-simulator

Prosthetic Arm BCI Simulator — a Next.js 15 dashboard for the ORBIT AI brain-computer interface system. Visualizes EEG-decoded motor imagery commands as robotic arm movements.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3001](http://localhost:3001) in your browser.

## Backend

Start the EEG stream and websocket server:

```bash
# Terminal 1: Start the EEG simulator
python simulate_tgam.py

# Terminal 2: Start the WebSocket server
python predict_websocket.py --demo
```

The dashboard connects to `ws://localhost:8765` for real-time EEG predictions.
