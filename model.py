import torch
import torch.nn as nn
from typing import Optional

class SelfAttention(nn.Module):
    """
    Self-attention mechanism to learn the importance of different timesteps.
    Used in the LSTM-based classifier.
    """
    def __init__(self, hidden_size: int):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output shape: [batch, seq_len, hidden_size]
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        # weights shape: [batch, seq_len, 1]
        context_vector = torch.sum(weights * lstm_output, dim=1)
        # context_vector shape: [batch, hidden_size]
        return context_vector

class EEGClassifier_LSTM(nn.Module):
    """
    LSTM-based EEG signal classifier with a self-attention mechanism.
    """
    def __init__(self, input_size: int = 18, hidden_size: int = 128, n_classes: int = 5):
        super(EEGClassifier_LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )
        
        # Bidirectional doubling hidden_size (128 * 2 = 256)
        self.attention = SelfAttention(hidden_size * 2)
        
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout_main = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 30, 18]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, 30, 256]
        
        context = self.attention(lstm_out)
        # context: [batch, 256]
        
        x = self.bn(context)
        x = self.dropout_main(x)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        logits = self.classifier(x)
        return logits

class EEGClassifier_CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for EEG classification.
    Processes features temporally with CNN first then LSTM.
    """
    def __init__(self, input_size: int = 18, n_classes: int = 5):
        super(EEGClassifier_CNN_LSTM, self).__init__()
        
        # CNN Block 1
        # Input: [batch, 18, 30] -> Conv produces [batch, 64, 30] -> Pool produces [batch, 64, 15]
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # CNN Block 2
        # Input: [batch, 64, 15] -> Conv produces [batch, 128, 15]
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        # LSTM
        # Input for LSTM: [batch, 15, 128]
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )
        
        # Classifier Head
        self.head_bn = nn.BatchNorm1d(128) # 64 * 2
        self.head_dropout1 = nn.Dropout(0.4)
        self.fc_final = nn.Linear(128, 64)
        self.relu_final = nn.ReLU()
        self.head_dropout2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 30, 18]
        
        # Reshape for Conv1D: [batch, channels, length]
        x = x.transpose(1, 2) # [batch, 18, 30]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x) # [batch, 64, 15]
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x) # [batch, 128, 15]
        
        # Reshape for LSTM: [batch, length, channels]
        x = x.transpose(1, 2) # [batch, 15, 128]
        
        lstm_out, _ = self.lstm(x) # [batch, 15, 128]
        
        # Global Average Pooling over time dimension
        # x.mean(dim=1) -> [batch, 128]
        x = torch.mean(lstm_out, dim=1)
        
        x = self.head_bn(x)
        x = self.head_dropout1(x)
        x = self.fc_final(x)
        x = self.relu_final(x)
        x = self.head_dropout2(x)
        logits = self.classifier(x)
        
        return logits

def get_model(use_cnn_lstm: bool = True, input_size: int = 18, n_classes: int = 5) -> nn.Module:
    """Factory function to build selected model."""
    if use_cnn_lstm:
        return EEGClassifier_CNN_LSTM(input_size=input_size, n_classes=n_classes)
    else:
        return EEGClassifier_LSTM(input_size=input_size, n_classes=n_classes)
