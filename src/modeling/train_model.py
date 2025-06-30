import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
from pathlib import Path

def load_and_prepare_data():
    """Load the data and split into features and target."""
    # Load data
    data_path = Path(__file__).parents[2] / 'data' / 'processed' / 'input_coordinates.xlsx'
    df = pd.read_excel(data_path)
    
    # Split into features (X) and target (y)
    X = df[['x1', 'y1', 'x2', 'y2']].values
    y = df[['x_opt', 'y_opt', 'z_opt']].values
    
    return X, y

def normalize_and_split_data(X, y):
    """Normalize features and targets, then split into train/validation/test sets."""
    # Initialize and fit scalers - using RobustScaler for better handling of outliers
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # Remove any potential infinite values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Clip values to prevent extreme outliers
    X_scaled = np.clip(X_scaled, -10, 10)
    y_scaled = np.clip(y_scaled, -10, 10)
    
    # First split: separate test set (80/20 split)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Second split: separate train and validation from temp (87.5/12.5 split of remaining data)
    # This gives us approximately 70/10/20 split overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42
    )
    
    return (x_scaler, y_scaler), (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_model(input_dim):
    """Create the MLP model architecture."""
    # Use gradient clipping and a smaller learning rate
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    # Initialize with small random weights
    initializer = tf.keras.initializers.HeNormal()
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,),
              kernel_initializer=initializer,
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu',
              kernel_initializer=initializer,
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu',
              kernel_initializer=initializer,
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(32, activation='relu',
              kernel_initializer=initializer,
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(3, kernel_initializer=initializer)  # Output layer
    ])
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),  # More robust to outliers than MSE
        metrics=['mae']
    )
    
    return model

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Check for NaN values in metrics
        if logs is not None:
            for metric, value in logs.items():
                if np.isnan(value) or np.isinf(value):
                    print(f"\nWarning: {metric} is NaN/Inf. Setting to a large value to continue training.")
                    logs[metric] = 1e6

def train_model():
    """Main function to train and save the model."""
    print("Loading data...")
    X, y = load_and_prepare_data()
    
    print("Normalizing and splitting data...")
    (x_scaler, y_scaler), (X_train, y_train), (X_val, y_val), (X_test, y_test) = normalize_and_split_data(X, y)
    
    print("Building model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()
    
    # Early stopping callback with larger patience
    early_stopping = EarlyStopping(
        monitor='loss',  # Monitor training loss instead of validation
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',  # Monitor training loss instead of validation
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Custom callback to handle NaN values
    custom_callback = CustomCallback()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, custom_callback],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest set performance (normalized):")
    print(f"Huber Loss: {test_loss:.4f}")
    print(f"Mean Absolute Error: {test_mae:.4f}")
    
    # Calculate error in millimeters
    y_pred = model.predict(X_test, verbose=0)
    y_pred_mm = y_scaler.inverse_transform(y_pred)
    y_test_mm = y_scaler.inverse_transform(y_test)
    mae_mm = np.mean(np.abs(y_pred_mm - y_test_mm))
    mse_mm = np.mean(np.square(y_pred_mm - y_test_mm))
    
    print(f"\nTest set performance (in millimeters):")
    print(f"Mean Squared Error: {mse_mm:.4f}")
    print(f"Mean Absolute Error: {mae_mm:.4f}")
    
    # Save model and scalers
    models_dir = Path(__file__).parents[2] / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'mlp_converter.keras'
    x_scaler_path = models_dir / 'x_scaler.pkl'
    y_scaler_path = models_dir / 'y_scaler.pkl'
    
    print(f"\nSaving model to {model_path}")
    model.save(model_path)
    
    print(f"Saving scalers to {x_scaler_path} and {y_scaler_path}")
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

if __name__ == "__main__":
    train_model()
