import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config
from model import EEGClassifier

def evaluate_model():
    # 1. Load data
    print("Loading test data...")
    X_test = np.load(os.path.join(config.DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(config.DATA_DIR, "y_test.npy"))
    
    # 2. Initialize and Load model
    model = EEGClassifier().to(config.DEVICE)
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: {config.BEST_MODEL_PATH} not found. Train the model first.")
        return
        
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    print("Best model loaded successfully.")
    
    # 3. Model Inference
    X_tensor = torch.from_numpy(X_test).float().to(config.DEVICE)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        
    # 4. Overall Accuracy
    accuracy = np.mean(preds == y_test)
    print(f"\n--- Final Evaluation ---")
    print(f"Overall Accuracy: {accuracy:.4%}")
    
    # 5. Classification Report
    print("\nClassification Report:")
    target_names = list(config.CLASSES.values())
    print(classification_report(y_test, preds, target_names=target_names))
    
    # 6. Confusion Matrix Heatmap
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('EEG Classifier: Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix heatmap saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
