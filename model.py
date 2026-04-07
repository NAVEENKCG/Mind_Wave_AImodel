import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class AttentionMechanism(nn.Module):
    """
    Computes a weighted average of sequence hidden states.
    """
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        weights = self.attention(x) # [batch, seq_len, 1]
        weights = F.softmax(weights, dim=1)
        
        # Multiply weights by elements
        context = torch.sum(x * weights, dim=1) # [batch, hidden_dim]
        return context, weights

class EEGClassifier(nn.Module):
    """
    Advanced Time-Series Hybrid Model (CNN + Bi-LSTM + Attention)
    Optimized for EEG signal classification.
    """
    def __init__(self, use_cnn_lstm=config.USE_CNN_LSTM):
        super(EEGClassifier, self).__init__()
        self.use_cnn_lstm = use_cnn_lstm
        
        # --- CNN Feature Extraction ---
        # Input shape: [batch, num_features, seq_len]
        self.conv_block = nn.Sequential(
            nn.Conv1d(config.NUM_FEATURES, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) if config.WINDOW_SIZE > 4 else nn.Identity(),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # --- Recurrent Context Learning ---
        # The LSTM input size will depend on the output depth of the CNN block
        lstm_input_size = 64 if self.use_cnn_lstm else config.NUM_FEATURES
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.HIDDEN_SIZE_1,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if 2 > 1 else 0
        )
        
        # Bidirectional means output is hidden_size * 2
        lstm_out_dim = config.HIDDEN_SIZE_1 * 2
        
        # --- Attention Layer ---
        self.attention = AttentionMechanism(lstm_out_dim)
        
        # --- Final Classification ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, config.HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE_2, config.NUM_CLASSES)
        )
            
    def forward(self, x):
        """
        x: [batch, seq_len, features]
        """
        if self.use_cnn_lstm:
            # Transpose for Conv1d: [batch, features, seq_len]
            x = x.transpose(1, 2)
            x = self.conv_block(x)
            # Transpose back for LSTM: [batch, new_seq_len, features]
            x = x.transpose(1, 2)
            
        # LSTM layer
        # out: [batch, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)
        
        # Apply Attention instead of just taking the last state
        context, attn_weights = self.attention(lstm_out)
        
        # Classification
        logits = self.classifier(context)
        
        return logits

if __name__ == "__main__":
    # Local config simulation for standalone testing
    if not hasattr(config, 'NUM_FEATURES'):
        config.NUM_FEATURES = 11
        config.NUM_CLASSES = 5
        config.WINDOW_SIZE = 10
        config.HIDDEN_SIZE_1 = 64
        config.HIDDEN_SIZE_2 = 32
        config.DROPOUT = 0.3
        config.USE_CNN_LSTM = True

    # Test model with dummy input
    test_input = torch.randn(2, 10, 11) # [batch, seq_len, features]
    
    print("Testing Advanced CNN-LSTM-Attention Model:")
    model = EEGClassifier(use_cnn_lstm=True)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") 
    print("Success!")
