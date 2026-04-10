import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from config import *
from model import EEGClassifier_CNN_LSTM

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 1. Load Data
    try:
        X = np.load(DATA_DIR / "X_pretrained.npy")
        y = np.load(DATA_DIR / "y_pretrained.npy")
        print(f"📊 Loaded {len(X)} samples  |  Shape: {X.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Remap to binary: 0=IDLE, 1=FORWARD (ignore other classes)
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]

    # Balance classes by undersampling majority class
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    min_count = min(len(idx_0), len(idx_1))
    np.random.seed(42)
    idx_0 = np.random.choice(idx_0, min_count, replace=False)
    idx_1 = np.random.choice(idx_1, min_count, replace=False)
    idx_all = np.concatenate([idx_0, idx_1])
    np.random.shuffle(idx_all)
    X, y = X[idx_all], y[idx_all]
    print(f"   Balanced: {np.bincount(y)} (total {len(y)})")

    # Pad 11 features → 18 to match model architecture
    if X.shape[2] < 18:
        pad_width = 18 - X.shape[2]
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        print(f"   Padded features: {X.shape[2]}")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 2. Train/Val split (80/20)
    dataset = TensorDataset(X_tensor, y_tensor)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
    print(f"   Train: {n_train}  |  Val: {n_val}")

    # 3. Model + Optimizer + Scheduler
    model = EEGClassifier_CNN_LSTM(input_size=18, n_classes=2).to(device)
    
    # Class-weighted loss to handle imbalance
    class_counts = np.bincount(y, minlength=2).astype(float)
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # 4. Training Loop
    MODELS_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0
    EPOCHS = 50

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>7} | {'LR':>10}")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += by.size(0)
            correct += (pred == by).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc  = 100 * correct / total

        # --- Validate ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                _, pred = torch.max(out, 1)
                v_total += by.size(0)
                v_correct += (pred == by).sum().item()
        val_acc = 100 * v_correct / v_total

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  {epoch+1:>4}  | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_acc:>6.2f}% | {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        scheduler.step()

    print(f"{'='*60}")
    print(f"✅ Training Complete!  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
