"""
plot_results.py

This script creates a visualization of the ML model's performance in converting
fisheye camera (2D YOLO) coordinates to 3D OptiTrack coordinates. It generates a 3d trajectory plot plot
to analyze error distributions.

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
        errors.extend([(f'{coord}', e) for e in error])

    error_df = pd.DataFrame(errors, columns=['coordinate', 'Absolute Error'])

    fig, ax = plt.subplots(figsize=(6.5, 3.9)) # Updated for latex
    
    ax.set_axisbelow(True)

    sns.violinplot(data=error_df, x='coordinate', y='Absolute Error', inner=None, ax=ax, saturation=1)
    # plt.title('Error distribution by coordinate')
    plt.ylabel('absolute error~(mm)')
    plt.grid(True)

    plt.savefig(save_dir / 'Figure4_Error_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to orchestrate the loading of data and generation of all plots.
    """
    print("Loading predictions...")
    df = load_predictions()

    plots_dir = Path('./') # Save in same dir
    plots_dir.mkdir(exist_ok=True)

    print("Creating error distribution plot...")
    create_error_distribution_plot(df, plots_dir)

    print(f"\Plots have been saved to {plots_dir}")
    print("\nPlot description:")
    print("Figure4_Error_Distribution.png: Error distribution across coordinates")

if __name__ == "__main__":
    main()