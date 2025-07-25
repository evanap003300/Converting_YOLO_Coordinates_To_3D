import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import joblib

def load_model_and_scalers():
    models_dir = Path(__file__).parents[2] / 'models'
    
    # Load model
    model_path = models_dir / 'mlp_converter.keras'
    model = tf.keras.models.load_model(model_path)
    
    # Load scalers
    x_scaler = joblib.load(models_dir / 'x_scaler.pkl')
    y_scaler = joblib.load(models_dir / 'y_scaler.pkl')
    
    return model, x_scaler, y_scaler

def load_data():
    # Load the input coordinates file
    data_path = Path(__file__).parents[2] / 'data' / 'processed' / 'input_coordinates.xlsx'
    return pd.read_excel(data_path)

def make_predictions(model, x_scaler, y_scaler, df):
    # Extract features
    X = df[['x1', 'y1', 'x2', 'y2']].values
    
    # Normalize input
    X_scaled = x_scaler.transform(X)
    
    # Make predictions
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    return y_pred

def create_output_dataframe(df, predictions):
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

def calculate_errors(df):
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
    predictions_dir = Path(__file__).parents[2] / 'data' / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    # Save predictions
    output_path = predictions_dir / 'mlp_predictions.xlsx'
    print(f"\nSaving predictions to {output_path}")
    output_df.to_excel(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
