import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, USE_CNN_LSTM, CLASS_NAMES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    try:
        X_test = torch.FloatTensor(np.load(DATA_DIR / "X_test.npy"))
        y_test = np.load(DATA_DIR / "y_test.npy")
    except FileNotFoundError:
        logger.error("Test data not found.")
        return

    # Load model
    from model import get_model
    model = get_model(USE_CNN_LSTM).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=device))
    model.eval()

    # Inference
    all_preds = []
    all_probs = []
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        logits = model(X_test_dev)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_preds = preds.cpu().numpy()
        all_probs = probs.cpu().numpy()

    # Classification Report
    target_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    report = classification_report(y_test, all_preds, target_names=target_names)
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    logger.info(f"Confusion matrix saved to {RESULTS_DIR}")

    # Confidence Distribution
    plt.figure(figsize=(10, 6))
    confidences = np.max(all_probs, axis=1)
    plt.hist(confidences, bins=30, color='skyblue', edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.savefig(RESULTS_DIR / "confidence_distribution.png")

    # Simple Feature Importance (Permutation)
    # Shuffling features in time dimension and measuring drop in accuracy
    feature_importance = {}
    base_acc = (all_preds == y_test).mean()
    
    # Feature labels for plotting
    feature_labels = [
        'delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma',
        'attention', 'meditation', 'blink', 'theta_beta_ratio', 'alpha_beta_ratio', 
        'attn_med_diff', 'total_power', 'beta_ratio', 'alpha_ratio', 'theta_ratio'
    ]

    for i in range(X_test.shape[2]):
        X_test_shuffled = X_test.clone()
        # Shuffle across all samples and all timesteps for this feature
        flat_feat = X_test_shuffled[:, :, i].numpy().flatten()
        np.random.shuffle(flat_feat)
        X_test_shuffled[:, :, i] = torch.tensor(flat_feat.reshape(X_test.shape[0], X_test.shape[1]))
        
        with torch.no_grad():
            shuffled_logits = model(X_test_shuffled.to(device))
            shuffled_preds = torch.argmax(shuffled_logits, dim=1).cpu().numpy()
            shuffled_acc = (shuffled_preds == y_test).mean()
            feature_importance[feature_labels[i]] = base_acc - shuffled_acc

    # Plot Feature Importance
    plt.figure(figsize=(12, 6))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    plt.bar(sorted_importance.keys(), sorted_importance.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance (Permutation Importance)')
    plt.ylabel('Accuracy Drop')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png")

if __name__ == "__main__":
    evaluate_model()
