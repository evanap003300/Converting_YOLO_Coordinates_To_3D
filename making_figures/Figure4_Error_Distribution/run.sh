#!/bin/bash

# generate_figure.sh
#
# This script is designed to be run from within a 'making_figures/FigureX_DescriptiveName/' subfolder.
# It orchestrates the process of generating the specific figure by:
# 1. Assuming necessary data (input_coordinates.xlsx) and trained model/scalers are present.
# 2. Running the prediction script to generate mlp_predictions.xlsx based on local data.
# 3. Running the plotting script to generate the figure image.
#
# Author: Evan Phillips
# Date: 2025-07-25

set -e

echo "Starting figure generation script from $(pwd)..."

# --- Step 1: Verify presence of required input files ---
echo "Verifying presence of required input data and model files..."
required_files=(
    "input_coordinates.xlsx"
    "mlp_converter.keras"
    "x_scaler.pkl"
    "y_scaler.pkl"
    "predict_and_export.py"
    "plot_results.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file '$file' not found in the current directory."
        echo "Please ensure all necessary data, model, and script files are copied into this folder."
        exit 1
    fi
done

echo "All required input files found."

# --- Step 2: Run the prediction script ---
# This script will take input_coordinates.xlsx (from this folder)
# and use the model/scalers (from this folder) to generate mlp_predictions.xlsx
# (which will also be created in this folder).
echo "Running the prediction script (predict_and_export.py)..."
python3 predict_and_export.py

# Give it a moment, especially if I/O is heavy (optional, remove if not needed)
echo "Pausing briefly..."
sleep 1 # Pause for 1 second

# --- Step 3: Run the plotting script ---
# This script will use the mlp_predictions.xlsx (generated in this folder)
# to create the figure image (e.g., FigureX_DescriptiveName.png).
echo "Running the plotting script (plot_results.py)..."
python3 plot_results.py

echo "Script execution complete."
echo "Success!"