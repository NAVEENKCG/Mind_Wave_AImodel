# Future Ideas

## PalmPay - AI-Based Palm Vein Biometric Payment Authentication System

**Overview**: A prototype demonstrating palm-vein biometrics for authenticating users and authorizing payments via a simulated digital wallet. It combines computer vision, machine learning, and full-stack development, without touching real financial systems.

**Key Objectives**:
- Register users and capture palm images.
- Process and enhance vein patterns.
- Generate secure biometric embeddings.
- Authenticate users against stored templates (rejecting unknowns).
- Gate simulated payments behind successful authentication.
- Maintain a demo wallet and transaction history.

**Tech Stack**:
- Frontend: React + Vite, Tailwind CSS
- Backend: Python + FastAPI
- CV/ML: OpenCV, NumPy, PyTorch, scikit-learn
- DB/Security: SQLite, hashed passwords, encrypted templates

**Target Hardware**:
- Development: Windows + standard webcam.
- Production: Raspberry Pi + NIR camera.

**Future Scope**:
- Kiosk deployment, real NIR camera integration, multi-user scale-up, stronger anti-spoofing, offline auth, and tokenized payments.
