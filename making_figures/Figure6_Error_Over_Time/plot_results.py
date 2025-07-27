"""
plot_results.py

This script creates a visualization of the ML model's performance in converting
fisheye camera (2D YOLO) coordinates to 3D OptiTrack coordinates. It generates a 3d trajectory plot plot
to analyze prediction accuracy over time.

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

    plt.figure(figsize=(6.5, 3.25)) # Updated for latex 
    for coord in coords:
        plt.plot(df['time'], errors[coord], label=f'{coord.upper()} error')

    plt.xlabel('Time~(s)')
    plt.ylabel('Absolute error~(mm)')
    plt.title('Prediction error over time')
    plt.grid(True)
    plt.legend()

    plt.savefig(save_dir / 'Figure6_Error_Over_Time.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to orchestrate the loading of data and generation of all plots.
    """
    print("Loading predictions...")
    df = load_predictions()

    plots_dir = Path('./') # Save in same dir
    plots_dir.mkdir(exist_ok=True)

    print("Creating error over time plot...")
    create_error_over_time_plot(df, plots_dir)

    print(f"\nAll plots have been saved to {plots_dir}")
    print("Figure6_Error_Over_Time.png: Error variation across time")

if __name__ == "__main__":
    main()