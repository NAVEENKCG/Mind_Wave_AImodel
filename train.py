import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import config
from model import EEGClassifier
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model():
    # Load processed data
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(config.DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(config.DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(config.DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.DATA_DIR, "y_val.npy"))
    
    # Create DataLoaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = EEGClassifier().to(config.DEVICE)
    print(f"Model initialized on {config.DEVICE}")
    print(f"Architecture: {'CNN-LSTM' if config.USE_CNN_LSTM else 'Pure LSTM'}")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.SCHEDULER_PATIENCE)
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        train_correct = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for data, target in train_bar:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target).sum().item()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_correct += (predicted == target).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_dataset)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"New best model saved with Accuracy: {best_val_acc:.2%}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Early Stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
            
    # Save training curves
    plot_history(history)
    print("\nTraining Complete! Curves saved to training_curves.png")

def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')

if __name__ == "__main__":
    train_model()
