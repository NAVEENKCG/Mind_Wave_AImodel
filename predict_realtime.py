import torch
import joblib
import numpy as np
import time
import serial
import os
import config
from model import EEGClassifier
from collect_data import TGAMParser

def predict_realtime():
    # 1. Load Model and Scaler
    print("Loading model and scaler for real-time inference...")
    model = EEGClassifier().to(config.DEVICE)
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: {config.BEST_MODEL_PATH} not found. Please train first.")
        return
        
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    print("Best model loaded successfully.")
    
    scaler = joblib.load(config.SCALER_PATH)
    print("Scaler loaded successfully.")
    
    # 2. Connect to TGAM
    parser = TGAMParser(config.SERIAL_PORT, config.BAUD_RATE)
    
    # Optional: Connect to wheelchair ESP32 for command output
    wheelchair_ser = None
    if config.WHEELCHAIR_PORT:
        try:
            wheelchair_ser = serial.Serial(config.WHEELCHAIR_PORT, 9600, timeout=1)
            print(f"Successfully connected to Wheelchair on {config.WHEELCHAIR_PORT}")
        except Exception as e:
            print(f"Error connecting to Wheelchair: {e}")
            
    # 3. Prediction Loop
    # We maintain a rolling buffer (window) for samples
    buffer = []
    print("\n--- Starting Live Inference ---")
    print(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Read single sample from TGAM
            data_dict = parser.read_packet()
            if data_dict:
                # Prepare feature vector (maintain exact order as in config)
                features = np.array([[data_dict[f] for f in config.FEATURES]])
                
                # Add to buffer
                buffer.append(features[0])
                
                # Keep buffer at WINDOW_SIZE
                if len(buffer) > config.WINDOW_SIZE:
                    buffer.pop(0)
                    
                # Run inference if buffer is full
                if len(buffer) == config.WINDOW_SIZE:
                    # Convert buffer to numpy and scale
                    input_window = np.array(buffer) # [window_size, num_features]
                    input_window_scaled = scaler.transform(input_window)
                    
                    # Convert to tensor: [batch=1, timesteps, features]
                    input_tensor = torch.from_numpy(input_window_scaled).float().unsqueeze(0).to(config.DEVICE)
                    
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        max_prob = np.max(probabilities)
                        predicted_label = np.argmax(probabilities)
                        
                    # Output prediction
                    class_name = config.CLASSES[predicted_label]
                    
                    if max_prob >= config.CONFIDENCE_THRESHOLD:
                        print(f"Prediction: {class_name:10} | Confidence: {max_prob:.4f} | CMD: {class_name}")
                        
                        # Send command to wheelchair ESP32
                        if wheelchair_ser:
                            wheelchair_ser.write(f"{predicted_label}\n".encode())
                    else:
                        print(f"Prediction: {class_name:10} | Confidence: {max_prob:.4f} (Under Threshold)")
                        
            # Wait for next sample (TGAM sends data approx every 1 sec)
            time.sleep(0.5) 
            
    except KeyboardInterrupt:
        print("\nStopping real-time inference.")
        
    finally:
        if wheelchair_ser:
            wheelchair_ser.close()

if __name__ == "__main__":
    predict_realtime()
