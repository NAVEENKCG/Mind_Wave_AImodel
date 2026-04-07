import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import config

def create_sliding_windows(data, window_size, stride):
    """
    data: (samples, features + label)
    returns: (windows, window_size, features), (labels)
    """
    X = []
    y = []
    
    # We ignore the timestamp for windowing
    # data format: [delta, theta, ..., blink, label]
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    
    for i in range(0, len(data) - window_size + 1, stride):
        # Extract features for this window
        window_feature = features[i:i + window_size]
        # Extract label for the END of the window (most descriptive label for current state)
        window_label = labels[i + window_size - 1]
        
        X.append(window_feature)
        y.append(window_label)
        
    return np.array(X), np.array(y)

def preprocess_data():
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"Error: {config.RAW_DATA_PATH} not found. Collecting samples first.")
        return
        
    print(f"Loading raw data from {config.RAW_DATA_PATH}...")
    # Load and drop timestamp
    df = pd.read_csv(config.RAW_DATA_PATH).drop(columns=['timestamp'])
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 1. Normalize features (exclude label)
    print("Normalizing features using MinMaxScaler (0 to 1)...")
    scaler = MinMaxScaler()
    df[config.FEATURES] = scaler.fit_transform(df[config.FEATURES])
    
    # Save the scaler for real-time inference
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    
    # 2. Create sliding windows
    print(f"Creating sliding windows (size={config.WINDOW_SIZE}, stride={config.STRIDE})...")
    X, y = create_sliding_windows(df, config.WINDOW_SIZE, config.STRIDE)
    
    print(f"Shape after windowing: X={X.shape}, y={y.shape}")
    
    # 3. Split into Train, Val, Test
    # Initial split: 80% train, 20% validation+test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT), 
        random_state=config.SEED,
        stratify=y
    )
    
    # Second split: 50/50 of temp -> val and test (10% and 10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=config.SEED,
        stratify=y_temp
    )
    
    # 4. Save processed data
    print("Saving processed data as .npy files...")
    np.save(os.path.join(config.DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(config.DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.DATA_DIR, "y_test.npy"), y_test)
    
    print("\nPre-processing Complete!")
    print(f"Train/Val/Test Distribution: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print("\nClass distribution in train set:")
    print(pd.Series(y_train).value_counts(normalize=True))

if __name__ == "__main__":
    preprocess_data()
