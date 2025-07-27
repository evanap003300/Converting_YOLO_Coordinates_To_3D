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

def create_scatter_plots(df: pd.DataFrame, save_dir: Path):
    """
    Generates individual scatter plots for X, Y, and Z coordinates,
    including MAE and RMSE in the title.
    Saved as 'Figure5_Scatter_X.png', 'Figure5_Scatter_Y.png', 'Figure5_Scatter_Z.png'.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted coordinates.
        save_dir (Path): Directory where the plot images will be saved.
    """
    coords = ['x', 'y', 'z']
    
    single_plot_figsize = (6.5, 4.8)

    for coord in coords:
        fig, ax = plt.subplots(1, 1, figsize=single_plot_figsize) # Create a single subplot figure

        true_vals = df[f'{coord}_opt']
        pred_vals = df[f'pred_{coord}']

        mae = np.mean(np.abs(true_vals - pred_vals))
        mse = np.mean(np.square(true_vals - pred_vals))
        rmse = np.sqrt(mse)

        ax.scatter(true_vals, pred_vals, alpha=0.6, label='Prediction')

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        ax.set_xlabel(f'True {coord.upper()} (mm)')
        ax.set_ylabel(f'Predicted {coord.upper()} (mm)')
        
        ax.set_title(f'{coord.upper()} Coordinate\nMAE = {mae:.2f} mm, RMSE = {rmse:.2f} mm',
                     fontsize=plt.rcParams['axes.titlesize'])
        # --------------------------------------------------------
        
        ax.grid(True)
        # Place legend inside the plot, but in a corner to avoid data overlap
        ax.legend(loc='upper left', frameon=True) 

        plt.tight_layout() # Ensure tight layout for this single plot
        plt.savefig(save_dir / f'Figure5_Scatter_{coord.upper()}.png', dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory

def main():
    """
    Main function to orchestrate the loading of data and generation of all plots.
    """
    print("Loading predictions...")
    df = load_predictions()

    plots_dir = Path('./') # Save in same dir
    plots_dir.mkdir(exist_ok=True)

    print("Creating scatter plots...")
    create_scatter_plots(df, plots_dir)

    print(f"\nPlots have been saved to {plots_dir}")
    print("\nPlot description:")
    print("Figure5_Scatter_Plots_(Dim).png: Predicted vs true values for each coordinate")

if __name__ == "__main__":
    main()