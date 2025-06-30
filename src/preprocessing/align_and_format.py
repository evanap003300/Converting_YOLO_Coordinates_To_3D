import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path so we can import from modeling
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.convert import convert_pixles_to_mm

def load_data():
    """Load both input files."""
    data_dir = Path(__file__).parents[2] / 'data'
    
    # Load fisheye coordinates
    fisheye_path = data_dir / 'raw' / 'fisheye_coordinates.xlsx'
    fisheye_df = pd.read_excel(fisheye_path)
    
    # Load OptiTrack coordinates
    optitrack_path = data_dir / 'raw' / 'Final_Trimmed_OptiTrack_Coordinates.xlsx'
    optitrack_df = pd.read_excel(optitrack_path)
    
    return fisheye_df, optitrack_df

def convert_normalized_to_mm(row, img_width=2160, img_height=2160):
    """Convert normalized coordinates to mm using the provided conversion function."""
    # Convert normalized to pixel values
    px1 = row['x1'] * img_width
    py1 = row['y1'] * img_height
    px2 = row['x2'] * img_width
    py2 = row['y2'] * img_height
    
    # Convert to mm using the provided function
    x1_mm, y1_mm = convert_pixles_to_mm(px1, py1)
    x2_mm, y2_mm = convert_pixles_to_mm(px2, py2)
    
    return pd.Series({
        'x1': x1_mm,
        'y1': y1_mm,
        'x2': x2_mm,
        'y2': y2_mm
    })

def align_coordinates(fisheye_df, optitrack_df, time_tolerance=0.01):
    """Match timestamps and align coordinates within the specified tolerance."""
    # Create output dataframe to store matched coordinates
    aligned_data = []
    
    # Ensure Time columns are numeric
    fisheye_df['Time'] = pd.to_numeric(fisheye_df['Time'])
    optitrack_df['Time'] = pd.to_numeric(optitrack_df['Time'])
    
    # Convert normalized coordinates to mm
    mm_coords = fisheye_df.apply(convert_normalized_to_mm, axis=1)
    fisheye_df[['x1', 'y1', 'x2', 'y2']] = mm_coords
    
    # For each fisheye timestamp, find the closest OptiTrack timestamp within tolerance
    for _, fisheye_row in fisheye_df.iterrows():
        time_diff = abs(optitrack_df['Time'] - fisheye_row['Time'])
        closest_idx = time_diff.idxmin()
        
        if time_diff[closest_idx] <= time_tolerance:
            optitrack_row = optitrack_df.loc[closest_idx]
            
            aligned_data.append({
                'Time': fisheye_row['Time'],
                'x1': fisheye_row['x1'],
                'y1': fisheye_row['y1'],
                'x2': fisheye_row['x2'],
                'y2': fisheye_row['y2'],
                'x_opt': optitrack_row['X'],
                'y_opt': optitrack_row['Y'],
                'z_opt': optitrack_row['Z']
            })
    
    return pd.DataFrame(aligned_data)

def main():
    # Load data
    print("Loading input files...")
    fisheye_df, optitrack_df = load_data()
    
    # Align coordinates
    print("Aligning coordinates...")
    aligned_df = align_coordinates(fisheye_df, optitrack_df)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parents[2] / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aligned coordinates
    output_path = output_dir / 'input_coordinates.xlsx'
    aligned_df.to_excel(output_path, index=False)
    print(f"Saved aligned coordinates to {output_path}")
    print(f"Total matched coordinate pairs: {len(aligned_df)}")

if __name__ == "__main__":
    main()
