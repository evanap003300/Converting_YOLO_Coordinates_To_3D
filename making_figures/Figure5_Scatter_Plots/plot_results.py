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
    Generates a combined figure with three subplots (x, y, z) showing predicted vs. true values.
    Saved as 'Figure5_Scatter.png'.
    """
    coords = ['x', 'y', 'z']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharex=False, sharey=False)
    
    for i, coord in enumerate(coords):
        ax = axes[i]
        ax.set_axisbelow(True)

        true_vals = df[f'{coord}_opt']
        pred_vals = df[f'pred_{coord}']

        mae = np.mean(np.abs(true_vals - pred_vals))
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))

        ax.scatter(true_vals, pred_vals, alpha=0.6, label='prediction', s=10)
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        ax.set_xlabel(rf'true {coord}~(mm)')
        ax.set_ylabel(rf'predicted {coord}~(mm)')  # Label all y-axes

        ax.grid(True)
        
        # Subfigure label (a), (b), (c)
        ax.text(0.02, 0.95, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=plt.rcParams['font.size'], va='top', ha='left')

    plt.tight_layout(w_pad=2)
    plt.savefig(save_dir / 'Figure5_Scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    print("Figure5_Scatter.png: Predicted vs true values for each coordinate")

if __name__ == "__main__":
    main()