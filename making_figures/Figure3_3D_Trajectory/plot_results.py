"""
plot_results.py

This script creates a visualization of the ML model's performance in converting
fisheye camera (2D YOLO) coordinates to 3D OptiTrack coordinates. It generates a 3d trajectory plot plot
to analyze prediction accuracy.

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
    'font.size': 8, 
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

def create_3d_trajectory_plot(df: pd.DataFrame, save_dir: Path):
    """
    Generates a 3D trajectory comparison plot of true vs. predicted paths.
    Saved as 'Figure3_3D_Trajectory.png'. Also generates multi-angle views.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted coordinates.
        save_dir (Path): Directory where the plot images will be saved.
    """
    fig = plt.figure(figsize=(8.0, 5.0))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(df['x_opt'], df['y_opt'], df['z_opt'],
            label='True trajectory', linewidth=2)
    ax.plot(df['pred_x'], df['pred_y'], df['pred_z'],
            label='Predicted trajectory', linewidth=2, linestyle='--')

    ax.scatter(df['x_opt'].iloc[0], df['y_opt'].iloc[0], df['z_opt'].iloc[0],
               color='green', s=100, label='Start')
    ax.scatter(df['x_opt'].iloc[-1], df['y_opt'].iloc[-1], df['z_opt'].iloc[-1],
               color='red', s=100, label='End')

    ax.set_xlabel('X~(mm)', fontsize=plt.rcParams['axes.labelsize'], labelpad=5) 
    ax.set_ylabel('Y~(mm)', fontsize=plt.rcParams['axes.labelsize'], labelpad=5) 
    ax.set_zlabel('Z~(mm)', fontsize=plt.rcParams['axes.labelsize'], labelpad=5) 
    
    # Set title with padding
    ax.set_title('3D trajectory comparison', fontsize=plt.rcParams['axes.titlesize'], pad=15)
    
    # Set legend font size
    ax.legend(fontsize=plt.rcParams['legend.fontsize'])

    # X-axis
    ax.tick_params(axis='x', pad=2, labelsize=plt.rcParams['xtick.labelsize'])
    ax.xaxis.set_tick_params(rotation=20) 
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    # Y-axis
    ax.tick_params(axis='y', pad=2, labelsize=plt.rcParams['ytick.labelsize'])
    ax.yaxis.set_tick_params(rotation=-20) 
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    # Z-axis
    ax.tick_params(axis='z', pad=2, labelsize=plt.rcParams.get('ztick.labelsize', plt.rcParams['ytick.labelsize']))
    ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    ax.view_init(elev=25, azim=315) 

    # Use tight_layout with a 'rect' to control the boundaries of the plot area within the figure.
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95]) 
    
    plt.savefig(save_dir / 'Figure3_3D_Trajectory.png', dpi=300)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(save_dir / 'Figure3_3D_Trajectory.png', dpi=300)

    plt.close(fig) # Close the figure

def main():
    """
    Main function to orchestrate the loading of data and generation of all plots.
    """
    print("Loading predictions...")
    df = load_predictions()

    plots_dir = Path('./') # Save in same dir
    plots_dir.mkdir(exist_ok=True)

    print("Creating 3D trajectory plot...")
    create_3d_trajectory_plot(df, plots_dir)

    print(f"\Plots have been saved to {plots_dir}")
    print("\nPlot description:")
    print("Figure3_3D_Trajectory.png: 3D trajectory comparison (true vs predicted)")

if __name__ == "__main__":
    main()