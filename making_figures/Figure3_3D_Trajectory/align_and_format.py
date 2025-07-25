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
    fisheye_path = data_dir / 'raw' / 'fisheye_coordinates.xlsx' # change the name of the file here 
    fisheye_df = pd.read_excel(fisheye_path)
    
    
    return fisheye_df

# image width and height may be wrong here 
# old was 2160 by 2160 
def convert_normalized_to_mm(row, img_width=3840, img_height=2160):
    """Convert normalized coordinates to mm using the provided conversion function."""
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
    # Hardcoded input and output paths
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
