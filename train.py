import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, USE_CNN_LSTM, SEED, N_CLASSES
)
from model import get_model

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_class_weights(y_train):
    """Compute weights for balanced loss."""
    counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    try:
        X_train = torch.FloatTensor(np.load(DATA_DIR / "X_train.npy"))
        y_train = torch.LongTensor(np.load(DATA_DIR / "y_train.npy"))
        X_val = torch.FloatTensor(np.load(DATA_DIR / "X_val.npy"))
        y_val = torch.LongTensor(np.load(DATA_DIR / "y_val.npy"))
    except FileNotFoundError:
        logger.error("Training data not found. Run preprocess.py first.")
        return

    # DataLoaders
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Initialize model
    model = get_model(USE_CNN_LSTM).to(device)
    logger.info(f"Initialized {'CNN-LSTM' if USE_CNN_LSTM else 'LSTM'} Model")

    # Loss, Optimizer, Scheduler
    class_weights = calculate_class_weights(y_train.numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    # Tracking
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})

        # Scheduler step (per epoch)
        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()

        # Calc statistics
        t_acc = train_correct / len(X_train)
        v_acc = val_correct / len(X_val)
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        curr_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Train Acc: {t_acc:.2%} | Val Acc: {v_acc:.2%} | LR: {curr_lr:.6f}")

        # Save history
        history.append({
            'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss,
            'train_acc': t_acc, 'val_acc': v_acc, 'lr': curr_lr,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Checkpointing
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            epochs_no_improve = 0
            logger.info("  *** Best model saved! ***")
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), MODELS_DIR / f"checkpoint_epoch_{epoch}.pth")

        # Early Stopping
        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    # Save final logs
    pd.DataFrame(history).to_csv(LOGS_DIR / "training_log.csv", index=False)
    
    # Plotting
    plot_curves(history)

def plot_curves(history):
    df = pd.DataFrame(history)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train')
    plt.plot(df['epoch'], df['val_loss'], label='Val')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train')
    plt.plot(df['epoch'], df['val_acc'], label='Val')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['lr'])
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png")
    logger.info(f"Training curves saved to {RESULTS_DIR}")

if __name__ == "__main__":
    train_model()
