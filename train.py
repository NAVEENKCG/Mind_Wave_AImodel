import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from config import *
from model import get_model, EEGClassifier_CNN_LSTM

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on: {device}")
    
    # 1. Load Data
    try:
        X = np.load(DATA_DIR / "X_pretrained.npy")
        y = np.load(DATA_DIR / "y_pretrained.npy")
        print(f"📊 Loaded {len(X)} clinical samples for training.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Convert to Tensors and Pad to 18 features (matches predict_realtime.py)
    # X shape: [1098, 30, 11] -> [1098, 30, 18]
    X_padded = np.zeros((X.shape[0], X.shape[1], 18))
    X_padded[:, :, :11] = X
    
    X_tensor = torch.FloatTensor(X_padded)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model (18 features)
    model = EEGClassifier_CNN_LSTM(input_size=18, n_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    MODELS_DIR.mkdir(exist_ok=True)
    best_loss = float('inf')
    
    print("\n--- Training Progress ---")
    for epoch in range(10): # Quick 10 epochs for the demo
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/10] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"\n✅ Training Complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
