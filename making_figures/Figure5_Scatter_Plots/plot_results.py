"""
plot_results.py

This script creates comprehensive visualizations of the ML model's performance in converting
fisheye camera (2D YOLO) coordinates to 3D OptiTrack coordinates. It generates various plots
to analyze prediction accuracy, error distributions, and trajectory comparisons, adhering
to specified lab style guidelines for publication.

Author: Evan Phillips
Date: 2025-07-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import seaborn as sns

# ====== Lab Style Guide Compliance: Plot Settings ======
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif',
        'Computer Modern Roman', 'New Century Schoolbook',
        'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman',
        'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif'
    ],
    'font.size': 10,
    'mathtext.rm': 'serif',
    'mathtext.fontset': 'custom',
    'image.cmap': 'viridis'
})
sns.set_palette("tab10")  # For categorical variables like coordinates
plt.close('all')
# ========================================================

def load_predictions() -> pd.DataFrame:
    """
    Loads the MLP model's 3D coordinate predictions from an Excel file.

    Args:
        base_path (Path): The base path of the project (e.g., the root of the GitHub repo clone).

    Returns:
        pd.DataFrame: DataFrame containing true and predicted 3D coordinates.
    """
    # data_path = Path(__file__).parents[2] / 'data' / 'predictions' / 'mlp_predictions.xlsx'
    data_path = Path('./') / 'mlp_predictions.xlsx'
    return pd.read_excel(data_path)

def create_scatter_plots(df: pd.DataFrame, save_dir: Path):
    """
    Generates scatter plots comparing true vs. predicted coordinates for X, Y, and Z.
    Saved as 'Figure5_Scatter_Plots.png'.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted coordinates.
        save_dir (Path): Directory where the plot images will be saved.
    """
    coords = ['x', 'y', 'z']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, coord in enumerate(coords):
        true_vals = df[f'{coord}_opt']
        pred_vals = df[f'pred_{coord}']

        mae = np.mean(np.abs(true_vals - pred_vals))
        mse = np.mean(np.square(true_vals - pred_vals))
        rmse = np.sqrt(mse)

        axes[i].scatter(true_vals, pred_vals, alpha=0.5, label='Prediction')

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        axes[i].set_xlabel(f'True {coord.upper()}~(mm)')
        axes[i].set_ylabel(f'Predicted {coord.upper()}~(mm)')
        axes[i].set_title(f'{coord.upper()} coordinate: MAE = {mae:.2f}~mm, RMSE = {rmse:.2f}~mm')
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'Figure5_Scatter_Plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_trajectory_plot(df: pd.DataFrame, save_dir: Path):
    """
    Generates a 3D trajectory comparison plot of true vs. predicted paths.
    Saved as 'Figure3_3D_Trajectory.png'. Also generates multi-angle views.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted coordinates.
        save_dir (Path): Directory where the plot images will be saved.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(df['x_opt'], df['y_opt'], df['z_opt'],
            label='True trajectory', linewidth=2)
    ax.plot(df['pred_x'], df['pred_y'], df['pred_z'],
            label='Predicted trajectory', linewidth=2, linestyle='--')

    ax.scatter(df['x_opt'].iloc[0], df['y_opt'].iloc[0], df['z_opt'].iloc[0],
               color='green', s=100, label='Start')
    ax.scatter(df['x_opt'].iloc[-1], df['y_opt'].iloc[-1], df['z_opt'].iloc[-1],
               color='red', s=100, label='End')

    ax.set_xlabel('X~(mm)')
    ax.set_ylabel('Y~(mm)')
    ax.set_zlabel('Z~(mm)')
    ax.set_title('3D trajectory comparison')
    ax.legend()

    plt.savefig(save_dir / 'Figure3_3D_Trajectory.png', dpi=300, bbox_inches='tight')

    for angle in range(0, 360, 45):
        ax.view_init(elev=20, azim=angle)
        plt.savefig(save_dir / f'3d_trajectory_angle_{angle}.png', dpi=300, bbox_inches='tight')

    plt.close()

def create_error_over_time_plot(df: pd.DataFrame, save_dir: Path):
    """
    Generates a plot showing prediction error over time for X, Y, and Z coordinates.
    Saved as 'Figure6_Error_Over_Time.png'.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted coordinates.
        save_dir (Path): Directory where the plot images will be saved.
    """
    coords = ['x', 'y', 'z']
    errors = pd.DataFrame()

    for coord in coords:
        errors[coord] = np.abs(df[f'{coord}_opt'] - df[f'pred_{coord}'])

    plt.figure(figsize=(12, 6))
    for coord in coords:
        plt.plot(df['time'], errors[coord], label=f'{coord.upper()} error')

    plt.xlabel('Time~(s)')
    plt.ylabel('Absolute error~(mm)')
    plt.title('Prediction error over time')
    plt.grid(True)
    plt.legend()

    plt.savefig(save_dir / 'Figure6_Error_Over_Time.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_distribution_plot(df: pd.DataFrame, save_dir: Path):
    """
    Generates a violin plot visualizing the distribution of absolute errors by coordinate axis.
    Saved as 'Figure4_Error_Distribution.png'.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted coordinates.
        save_dir (Path): Directory where the plot images will be saved.
    """
    coords = ['x', 'y', 'z']
    errors = []

    for coord in coords:
        error = np.abs(df[f'{coord}_opt'] - df[f'pred_{coord}'])
        errors.extend([(coord.upper(), e) for e in error])

    error_df = pd.DataFrame(errors, columns=['Coordinate', 'Absolute Error'])

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=error_df, x='Coordinate', y='Absolute Error')
    plt.title('Error distribution by coordinate')
    plt.ylabel('Absolute error~(mm)')
    plt.grid(True)

    plt.savefig(save_dir / 'Figure4_Error_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to orchestrate the loading of data and generation of all plots.
    """
    print("Loading predictions...")
    df = load_predictions()

    # plots_dir = Path(__file__).parents[2] / 'data' / 'plots'
    plots_dir = Path('./') # Save in same dir
    plots_dir.mkdir(exist_ok=True)

    print("Creating scatter plots...")
    create_scatter_plots(df, plots_dir)

    #print("Creating 3D trajectory plot...")
    #create_3d_trajectory_plot(df, plots_dir)

    #print("Creating error over time plot...")
    #create_error_over_time_plot(df, plots_dir)

    #print("Creating error distribution plot...")
    #create_error_distribution_plot(df, plots_dir)

    print(f"\nAll plots have been saved to {plots_dir}")
    print("\nPlot descriptions:")
    print("1. scatter_plots.png: Predicted vs true values for each coordinate")
    #print("2. 3d_trajectory.png: 3D trajectory comparison (true vs predicted)")
    #print("3. 3d_trajectory_angle_*.png: Multi-angle views of trajectory")
    #print("4. error_over_time.png: Error variation across time")
    #print("5. error_distribution.png: Error distribution across coordinates")

if __name__ == "__main__":
    main()