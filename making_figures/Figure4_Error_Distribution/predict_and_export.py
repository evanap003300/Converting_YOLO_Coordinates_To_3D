"""
predict_and_export.py

This script is responsible for loading a trained machine learning model and its
associated scalers, making 3D coordinate predictions based on processed input data,
and exporting the results. It also includes functions for evaluating model
performance by calculating various error metrics.

Author: Evan Phillips
Date: 2025-07-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import joblib

def load_model_and_scalers() -> tuple[tf.keras.Model, object, object]:
    """
    Loads the pre-trained Keras/TensorFlow model and the input/output scalers.

    Returns:
        tuple[tf.keras.Model, joblib.NumpyArrayFile, joblib.NumpyArrayFile]:
            A tuple containing:
            - model (tf.keras.Model): The loaded neural network model.
            - x_scaler (joblib.NumpyArrayFile): The scaler used for input features.
            - y_scaler (joblib.NumpyArrayFile): The scaler used for output (target) coordinates.
    """
    
    # models_dir = Path(__file__).parents[2] / 'models'
    models_dir = Path('./')  # Updated to allow for output in the same folder for figure replication
    
    # Load model
    model_path = models_dir / 'mlp_converter.keras'
    model = tf.keras.models.load_model(model_path)
    
    # Load scalers
    x_scaler = joblib.load(models_dir / 'x_scaler.pkl')
    y_scaler = joblib.load(models_dir / 'y_scaler.pkl')
    
    return model, x_scaler, y_scaler

def load_data() -> pd.DataFrame:
    """
    Loads the processed input coordinate data (e.g., from align_and_format.py)
    that will be fed into the machine learning model for prediction.

    Returns:
        pd.DataFrame: A DataFrame containing the input features ('x1', 'y1', 'x2', 'y2'),
                      time, and actual OptiTrack values ('x_opt', 'y_opt', 'z_opt').
    """

    # Load the input coordinates file
    # data_path = Path(__file__).parents[2] / 'data' / 'processed' / 'input_coordinates.xlsx'
    data_path = Path('./') / 'input_coordinates.xlsx' # Updated to work for files in this dir
    return pd.read_excel(data_path)

def make_predictions(model: tf.keras.Model, x_scaler: object, 
                     y_scaler: object, df: pd.DataFrame) -> np.ndarray:
    """
    Uses the loaded model and scalers to make 3D coordinate predictions.

    Args:
        model (tf.keras.Model): The trained neural network model.
        x_scaler (joblib.NumpyArrayFile): Scaler for input features.
        y_scaler (joblib.NumpyArrayFile): Scaler for output (target) coordinates.
        df (pd.DataFrame): DataFrame containing the input features ('x1', 'y1', 'x2', 'y2').

    Returns:
        np.ndarray: An array of inverse-transformed 3D predictions in original millimeter units.
                    Shape will be (num_samples, 3) for (x, y, z).
    """

    # Extract features
    X = df[['x1', 'y1', 'x2', 'y2']].values
    
    # Normalize input
    X_scaled = x_scaler.transform(X)
    
    # Make predictions
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    return y_pred

def create_output_dataframe(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Combines original data with new predictions into a single DataFrame for export and analysis.

    Args:
        df (pd.DataFrame): The original input DataFrame containing 'Time' and OptiTrack
                           'x_opt', 'y_opt', 'z_opt' columns.
        predictions (np.ndarray): The 3D predictions array from the model.

    Returns:
        pd.DataFrame: A new DataFrame structured for output, including time,
                      predicted coordinates, and actual OptiTrack coordinates.
    """
    output_df = pd.DataFrame()
    
    # Add time column
    output_df['time'] = df['Time']
    
    # Add predictions
    output_df['pred_x'] = predictions[:, 0]
    output_df['pred_y'] = predictions[:, 1]
    output_df['pred_z'] = predictions[:, 2]
    
    # Add actual values
    output_df['x_opt'] = df['x_opt']
    output_df['y_opt'] = df['y_opt']
    output_df['z_opt'] = df['z_opt']
    
    return output_df

def calculate_errors(df: pd.DataFrame):
    """
    Calculates and prints Mean Absolute Error (MAE), Mean Squared Error (MSE),
    and Root Mean Squared Error (RMSE) for each coordinate and overall.

    Args:
        df (pd.DataFrame): DataFrame containing both 'pred_x/y/z' and 'x_opt/y_opt/z_opt' columns.
    """

    pred_cols = ['pred_x', 'pred_y', 'pred_z']
    actual_cols = ['x_opt', 'y_opt', 'z_opt']
    
    # Calculate errors for each coordinate
    errors = {}
    for pred, actual in zip(pred_cols, actual_cols):
        diff = df[pred] - df[actual]
        mae = np.mean(np.abs(diff))
        mse = np.mean(np.square(diff))
        rmse = np.sqrt(mse)
        errors[actual] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    
    # Calculate overall error
    total_mae = np.mean([errors[col]['MAE'] for col in actual_cols])
    total_mse = np.mean([errors[col]['MSE'] for col in actual_cols])
    total_rmse = np.sqrt(total_mse)
    
    print("\nPrediction Errors (in millimeters):")
    print(f"Overall MAE: {total_mae:.2f}")
    print(f"Overall RMSE: {total_rmse:.2f}")
    print("\nPer-coordinate errors:")
    for coord in actual_cols:
        print(f"\n{coord}:")
        print(f"MAE: {errors[coord]['MAE']:.2f}")
        print(f"RMSE: {errors[coord]['RMSE']:.2f}")

def main():
    """
    Main function to orchestrate the prediction workflow:
    1. Load trained model and scalers.
    2. Load input data.
    3. Make predictions.
    4. Create output DataFrame.
    5. Calculate and display error metrics.
    6. Save the predictions to an Excel file.
    """
    
    print("Loading model and scalers...")
    model, x_scaler, y_scaler = load_model_and_scalers()
    
    print("Loading input data...")
    df = load_data()
    
    print("Making predictions...")
    predictions = make_predictions(model, x_scaler, y_scaler, df)
    
    print("Creating output dataframe...")
    output_df = create_output_dataframe(df, predictions)
    
    # Calculate and display errors
    calculate_errors(output_df)
    
    # Create predictions directory if it doesn't exist
    # predictions_dir = Path(__file__).parents[2] / 'data' / 'predictions'
    predictions_dir = Path('./') # Outputs in this dir
    predictions_dir.mkdir(exist_ok=True)
    
    # Save predictions
    output_path = predictions_dir / 'mlp_predictions.xlsx'
    print(f"\nSaving predictions to {output_path}")
    output_df.to_excel(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
