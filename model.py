import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class EEGClassifier(nn.Module):
    """
    A time-series classifier for EEG signals.
    Supports both a pure LSTM architecture and a CNN-LSTM hybrid.
    """
    def __init__(self, use_cnn_lstm=config.USE_CNN_LSTM):
        super(EEGClassifier, self).__init__()
        self.use_cnn_lstm = use_cnn_lstm
        
        # --- Shared Layers ---
        self.dropout = nn.Dropout(config.DROPOUT)
        
        if self.use_cnn_lstm:
            # --- CNN-LSTM Hybrid ---
            # Input: [batch, input_dim, timesteps] for 1D Conv
            self.conv1 = nn.Conv1d(in_channels=config.NUM_FEATURES, out_channels=32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=2)
            
            # After conv/pool (10 timesteps -> 5 timesteps)
            # Input for LSTM: [batch, timesteps=5, features=32]
            self.lstm = nn.LSTM(input_size=32, hidden_size=config.HIDDEN_SIZE_1, batch_first=True)
            self.fc = nn.Linear(config.HIDDEN_SIZE_1, config.NUM_CLASSES)
            
        else:
            # --- Pure LSTM Architecture ---
            # Input: [batch, timesteps=10, features=11]
            self.lstm1 = nn.LSTM(input_size=config.NUM_FEATURES, hidden_size=config.HIDDEN_SIZE_1, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=config.HIDDEN_SIZE_1, hidden_size=config.HIDDEN_SIZE_2, batch_first=True)
            self.fc = nn.Linear(config.HIDDEN_SIZE_2, config.NUM_CLASSES)
            
    def forward(self, x):
        """
        x: [batch, timesteps, features]
        """
        if self.use_cnn_lstm:
            # CNN expects (batch, channels, length)
            x = x.transpose(1, 2)
            x = self.pool(self.relu(self.conv1(x)))
            # Transpose back for LSTM: (batch, length, channels)
            x = x.transpose(1, 2)
            
            out, _ = self.lstm(x)
            # Use last hidden state
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            
        else:
            # Pure LSTM
            out, _ = self.lstm1(x)
            out, _ = self.lstm2(out)
            # Use last hidden state
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            
        return out

if __name__ == "__main__":
    # Test model with dummy input
    test_input = torch.randn(1, config.WINDOW_SIZE, config.NUM_FEATURES)
    
    print("Testing Pure LSTM Model:")
    model_lstm = EEGClassifier(use_cnn_lstm=False)
    output_lstm = model_lstm(test_input)
    print(f"Output shape: {output_lstm.shape}") # Expect [1, 5]
    
    print("\nTesting CNN-LSTM Model:")
    model_cnn = EEGClassifier(use_cnn_lstm=True)
    output_cnn = model_cnn(test_input)
    print(f"Output shape: {output_cnn.shape}") # Expect [1, 5]
