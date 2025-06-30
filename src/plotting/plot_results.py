import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import seaborn as sns

def load_predictions():
    """Load the predictions file."""
    data_path = Path(__file__).parents[2] / 'data' / 'predictions' / 'mlp_predictions.xlsx'
    return pd.read_excel(data_path)

def create_scatter_plots(df, save_dir):
    """Create scatter plots comparing predicted vs true values for each coordinate."""
    coords = ['x', 'y', 'z']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, coord in enumerate(coords):
        true_vals = df[f'{coord}_opt']
        pred_vals = df[f'pred_{coord}']
        
        # Calculate errors
        mae = np.mean(np.abs(true_vals - pred_vals))
        mse = np.mean(np.square(true_vals - pred_vals))
        rmse = np.sqrt(mse)
        
        # Create scatter plot
        axes[i].scatter(true_vals, pred_vals, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        axes[i].set_xlabel(f'True {coord.upper()} (mm)')
        axes[i].set_ylabel(f'Predicted {coord.upper()} (mm)')
        axes[i].set_title(f'{coord.upper()} Coordinate\nMAE: {mae:.2f}mm, RMSE: {rmse:.2f}mm')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_trajectory_plot(df, save_dir):
    """Create 3D plot showing predicted and true trajectories over time."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot true trajectory
    ax.plot(df['x_opt'], df['y_opt'], df['z_opt'], 
            label='True Trajectory', linewidth=2)
    
    # Plot predicted trajectory
    ax.plot(df['pred_x'], df['pred_y'], df['pred_z'], 
            label='Predicted Trajectory', linewidth=2, linestyle='--')
    
    # Add points to show direction
    ax.scatter(df['x_opt'].iloc[0], df['y_opt'].iloc[0], df['z_opt'].iloc[0], 
               color='green', s=100, label='Start')
    ax.scatter(df['x_opt'].iloc[-1], df['y_opt'].iloc[-1], df['z_opt'].iloc[-1], 
               color='red', s=100, label='End')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
    
    # Save static view
    plt.savefig(save_dir / '3d_trajectory.png', dpi=300, bbox_inches='tight')
    
    # Create multiple views
    for angle in range(0, 360, 45):
        ax.view_init(elev=20, azim=angle)
        plt.savefig(save_dir / f'3d_trajectory_angle_{angle}.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def create_error_over_time_plot(df, save_dir):
    """Create plot showing prediction error over time."""
    coords = ['x', 'y', 'z']
    errors = pd.DataFrame()
    
    # Calculate absolute errors for each coordinate
    for coord in coords:
        errors[coord] = np.abs(df[f'{coord}_opt'] - df[f'pred_{coord}'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    for coord in coords:
        plt.plot(df['time'], errors[coord], label=f'{coord.upper()} Error')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error (mm)')
    plt.title('Prediction Error Over Time')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(save_dir / 'error_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_distribution_plot(df, save_dir):
    """Create violin plots showing error distribution for each coordinate."""
    coords = ['x', 'y', 'z']
    errors = []
    
    # Calculate errors and prepare data for violin plot
    for coord in coords:
        error = np.abs(df[f'{coord}_opt'] - df[f'pred_{coord}'])
        errors.extend([(coord.upper(), e) for e in error])
    
    error_df = pd.DataFrame(errors, columns=['Coordinate', 'Absolute Error'])
    
    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=error_df, x='Coordinate', y='Absolute Error')
    plt.title('Error Distribution by Coordinate')
    plt.ylabel('Absolute Error (mm)')
    plt.grid(True)
    
    plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading predictions...")
    df = load_predictions()
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parents[2] / 'data' / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("Creating scatter plots...")
    create_scatter_plots(df, plots_dir)
    
    print("Creating 3D trajectory plot...")
    create_3d_trajectory_plot(df, plots_dir)
    
    print("Creating error over time plot...")
    create_error_over_time_plot(df, plots_dir)
    
    print("Creating error distribution plot...")
    create_error_distribution_plot(df, plots_dir)
    
    print(f"\nAll plots have been saved to {plots_dir}")
    print("\nPlot descriptions:")
    print("1. scatter_plots.png: Shows predicted vs true values for each coordinate")
    print("2. 3d_trajectory.png: Shows the complete 3D trajectory comparison")
    print("3. 3d_trajectory_angle_*.png: Multiple views of the 3D trajectory")
    print("4. error_over_time.png: Shows how prediction error varies over time")
    print("5. error_distribution.png: Shows the distribution of errors for each coordinate")

if __name__ == "__main__":
    main()
