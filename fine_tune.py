import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path

from config import (
    DATA_DIR, MODELS_DIR, USE_CNN_LSTM, SEED,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, N_CLASSES
)
from model import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pretrain_on_external_data():
    """
    Step 1: Train the model on large-scale OpenNeuro data.
    If labels aren't available, this could be Autoencoder-based (Self-Supervised).
    Here we assume we have labels for a general task.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        X_pre = np.load(DATA_DIR / "X_pretrained.npy")
        # For demo, creating dummy labels if they don't exist
        y_pre = np.random.randint(0, N_CLASSES, len(X_pre))
    except FileNotFoundError:
        logger.error("X_pretrained.npy not found. Run fetch_and_process_openneuro.py first.")
        return

    # Convert to windows if not already
    # X_pre is [samples, features] - we need [samples, windows, features]
    # Assuming the processing script already handles windowing or we do it here
    
    model = get_model(USE_CNN_LSTM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting Pre-training on OpenNeuro data...")
    # (Training loop omitted for brevity, similar to train.py)
    
    torch.save(model.state_dict(), MODELS_DIR / "openneuro_base_model.pth")
    logger.info("Base model saved to models/openneuro_base_model.pth")

def fine_tune_on_personal_data():
    """
    Step 2: Transfer general EEG knowledge to YOUR specific brain patterns.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load personal data
    X_train = torch.FloatTensor(np.load(DATA_DIR / "X_train.npy"))
    y_train = torch.LongTensor(np.load(DATA_DIR / "y_train.npy"))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # Load base model
    model = get_model(USE_CNN_LSTM).to(device)
    try:
        model.load_state_dict(torch.load(MODELS_DIR / "openneuro_base_model.pth"))
        logger.info("Successfully loaded OpenNeuro base model.")
    except FileNotFoundError:
        logger.warning("Base model not found. Training from scratch.")

    # FREEZE CNN LAYERS
    # We keep the lower-level feature extractors fixed
    if USE_CNN_LSTM:
        for param in model.conv1.parameters(): param.requires_grad = False
        for param in model.conv2.parameters(): param.requires_grad = False
    else:
        # For LSTM, freeze the first layer
        for i, (name, param) in enumerate(model.lstm.named_parameters()):
            if "l0" in name: # "l0" usually refers to the first layer of LSTM
                param.requires_grad = False

    # Fine-tuning uses a MUCH smaller learning rate
    fine_tune_lr = LEARNING_RATE * 0.1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=fine_tune_lr, weight_decay=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting Fine-tuning with LR={fine_tune_lr}...")
    
    for epoch in range(1, 21): # Fewer epochs for fine-tuning
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        logger.info(f"Fine-tune Epoch {epoch}/20 | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
    logger.info("Fine-tuned model saved as best_model.pth")

if __name__ == "__main__":
    # Uncomment to pre-train first
    # pretrain_on_external_data()
    
    fine_tune_on_personal_data()
