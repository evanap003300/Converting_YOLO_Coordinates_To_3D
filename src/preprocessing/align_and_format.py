"""
align_and_format.py

This script performs the initial data preprocessing for the drone tracking project.
It loads raw YOLO (fisheye camera) coordinate data, converts normalized pixel coordinates
to millimeters, and saves the processed data.

Author: Evan Phillips
Date: 2025-07-25
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_pixles_to_mm(x, y):
    new_x = x * (25.4/96)
    new_y = y * (25.4/96)
    return new_x, new_y

def convert_normalized_to_mm(row: pd.Series, img_width: int = 3840, img_height: int = 2160) -> pd.Series:
    """
    Applies the pixel-to-millimeter conversion to a single row of coordinate data.
    This function is designed to be used with DataFrame.apply().

    Args:
        row (pd.Series): A row from the DataFrame, expected to contain 'x1', 'y1', 'x2', 'y2'
                         representing normalized pixel coordinates.
        img_width (int): The width of the image in pixels (default: 3840).
        img_height (int): The height of the image in pixels (default: 2160).

    Returns:
        pd.Series: A Series containing the converted 'x1', 'y1', 'x2', 'y2' coordinates in millimeters.
    """
    px1 = row['x1'] 
    py1 = row['y1'] 
    px2 = row['x2'] 
    py2 = row['y2']
    
    # Convert to mm using the provided function
    x1_mm, y1_mm = convert_pixles_to_mm(px1, py1)
    x2_mm, y2_mm = convert_pixles_to_mm(px2, py2)
    
    return pd.Series({
        'x1': x1_mm,
        'y1': y1_mm,
        'x2': x2_mm,
        'y2': y2_mm
    })


def main():
    """
    Main function to orchestrate the data loading, coordinate conversion, and saving
    of the processed data.
    """
    input_path = Path(__file__).parents[2] / 'data' / 'raw' / 'YOLO_Coordinates.xlsx'
    output_path = Path(__file__).parents[2] / 'data' / 'processed' / 'YOLO_Coordinates_mm.xlsx'

    # Load data
    print(f"Loading input file: {input_path}")
    df = pd.read_excel(input_path)

    # Convert coordinates to mm
    print("Converting coordinates to mm...")
    converted_df = df.apply(convert_normalized_to_mm, axis=1)

    # Save converted coordinates
    converted_df.to_excel(output_path, index=False)
    print(f"Saved converted coordinates to {output_path}")
    print(f"Total coordinate pairs converted: {len(converted_df)}")

if __name__ == "__main__":
    main()
