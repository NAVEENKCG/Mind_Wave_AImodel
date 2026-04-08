import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from config import DATA_DIR, MODELS_DIR, SEED, TRAIN_WINDOW_SIZE, STRIDE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 derived features to the EEG dataframe."""
    epsilon = 1e-6
    
    # Total power calculation
    bands = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']
    df['total_power'] = df[bands].sum(axis=1)
    
    # Ratios
    df['theta_beta_ratio'] = df['theta'] / (df['lowBeta'] + df['highBeta'] + epsilon)
    df['alpha_beta_ratio'] = (df['lowAlpha'] + df['highAlpha']) / (df['lowBeta'] + df['highBeta'] + epsilon)
    df['attention_meditation_diff'] = df['attention'] - df['meditation']
    df['beta_ratio'] = (df['lowBeta'] + df['highBeta']) / (df['total_power'] + epsilon)
    df['alpha_ratio'] = (df['lowAlpha'] + df['highAlpha']) / (df['total_power'] + epsilon)
    df['theta_ratio'] = df['theta'] / (df['total_power'] + epsilon)
    
    return df

def augment_data(X: np.ndarray) -> np.ndarray:
    """Implement 4 types of data augmentation on a single window sample [30, 18]."""
    augmented_samples = []
    
    # Original
    augmented_samples.append(X)
    
    # 1. Gaussian Noise (3 versions)
    for _ in range(3):
        noise = np.random.normal(0, 0.02, X.shape)
        augmented_samples.append(X + noise)
        
    # 2. Amplitude Scaling (2 versions)
    for _ in range(2):
        scale = np.random.uniform(0.85, 1.15)
        augmented_samples.append(X * scale)
        
    # 3. Time Shift (2 versions)
    for _ in range(2):
        shift = np.random.randint(1, 5)
        augmented_samples.append(np.roll(X, shift, axis=0))
        
    # 4. Channel Dropout (1 version)
    dropout_sample = X.copy()
    indices = np.random.choice(X.shape[1], 2, replace=False)
    dropout_sample[:, indices] = 0
    augmented_samples.append(dropout_sample)
    
    return np.array(augmented_samples)

def create_sliding_windows(df: pd.DataFrame):
    """Convert dataframe into sliding window segments [samples, window_size, features]."""
    X, y = [], []
    
    # Assuming samples are grouped by session and sample_id from collection protocol
    groups = df.groupby(['session_id', 'sample_id'])
    
    for _, group in groups:
        # Extract feature columns (11 original + 7 derived)
        features = group.drop(['session_id', 'sample_id', 'timestep', 'label'], axis=1).values
        label = group['label'].iloc[0]
        
        # Apply sliding window within the sequence
        for i in range(0, len(features) - TRAIN_WINDOW_SIZE + 1, STRIDE):
            window = features[i : i + TRAIN_WINDOW_SIZE]
            if len(window) == TRAIN_WINDOW_SIZE:
                X.append(window)
                y.append(label)
                
    return np.array(X), np.array(y)

def run_preprocessing_pipeline():
    """Execute the full 8-step preprocessing pipeline."""
    raw_path = DATA_DIR / "raw_data.csv"
    if not raw_path.exists():
        logger.error("raw_data.csv not found. Run collect_data.py first.")
        return

    # STEP 1: LOAD AND VALIDATE
    df = pd.read_csv(raw_path)
    df = df.replace(0, np.nan).fillna(df.median(numeric_only=True)) # Minimal cleaning
    logger.info(f"Class distribution before cleaning: \n{df['label'].value_counts()}")

    # STEP 2: OUTLIER REMOVAL
    feature_cols = df.columns.difference(['session_id', 'sample_id', 'timestep', 'label'])
    before_count = len(df)
    for col in feature_cols:
        mean, std = df[col].mean(), df[col].std()
        df = df[df[col] <= (mean + 3 * std)]
    logger.info(f"Removed {before_count - len(df)} rows as outliers.")

    # STEP 4 (Applying early for augmentation): FEATURE ENGINEERING
    df = add_derived_features(df)
    
    # STEP 8: SPLIT BEFORE AUGMENTATION (Stratified)
    # We must split by 'sample_id' to ensure no leakage of augmented versions in val/test
    unique_ids = df[['session_id', 'sample_id', 'label']].drop_duplicates()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.3, stratify=unique_ids['label'], random_state=SEED)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, stratify=test_ids['label'], random_state=SEED)

    def filter_by_ids(full_df, ids_df):
        return full_df.merge(ids_df[['session_id', 'sample_id']], on=['session_id', 'sample_id'])

    train_df = filter_by_ids(df, train_ids)
    val_df = filter_by_ids(df, val_ids)
    test_df = filter_by_ids(df, test_ids)

    # STEP 5: SLIDING WINDOW CREATION
    X_train_raw, y_train_raw = create_sliding_windows(train_df)
    X_val, y_val = create_sliding_windows(val_df)
    X_test, y_test = create_sliding_windows(test_df)

    # STEP 3: NORMALIZATION (RobustScaler)
    scaler = RobustScaler()
    # Flatten to fit [samples*windows, features]
    original_shape = X_train_raw.shape
    X_train_flat = X_train_raw.reshape(-1, original_shape[2])
    scaler.fit(X_train_flat)
    
    def scale_dataset(X):
        s = X.shape
        return scaler.transform(X.reshape(-1, s[2])).reshape(s)

    X_train_scaled = scale_dataset(X_train_raw)
    X_val = scale_dataset(X_val)
    X_test = scale_dataset(X_test)
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # STEP 6: DATA AUGMENTATION (Train only)
    X_train_aug, y_train_aug = [], []
    for sample, label in zip(X_train_scaled, y_train_raw):
        augmented = augment_data(sample)
        X_train_aug.extend(augmented)
        y_train_aug.extend([label] * len(augmented))
    
    X_train = np.array(X_train_aug)
    y_train = np.array(y_train_aug)

    # STEP 7: CLASS BALANCING (Undersampling to minimum)
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    logger.info(f"Balancing classes to {min_count} samples each.")
    
    balanced_X, balanced_y = [], []
    for label in unique:
        idx = np.where(y_train == label)[0]
        np.random.seed(SEED)
        selected_idx = np.random.choice(idx, min_count, replace=False)
        balanced_X.append(X_train[selected_idx])
        balanced_y.append(y_train[selected_idx])
    
    X_train = np.vstack(balanced_X)
    y_train = np.hstack(balanced_y)

    # Final Save
    np.save(DATA_DIR / "X_train.npy", X_train)
    np.save(DATA_DIR / "y_train.npy", y_train)
    np.save(DATA_DIR / "X_val.npy", X_val)
    np.save(DATA_DIR / "y_val.npy", y_val)
    np.save(DATA_DIR / "X_test.npy", X_test)
    np.save(DATA_DIR / "y_test.npy", y_test)

    logger.info(f"Preprocessing complete. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
