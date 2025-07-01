# Drone Tracking with Fisheye Correction and 3D Coordinate Estimation

This project processes drone video footage from two fisheye cameras to track drone positions and convert them into 3D coordinates aligned with OptiTrack measurements. It uses machine learning to achieve accurate coordinate conversion and provides comprehensive visualization tools for analysis.

## Project Structure

```
task_1/
├── data/
│   ├── plots/          # Visualization outputs
│   ├── predictions/    # Model predictions
│   ├── processed/      # Processed coordinate data
│   └── raw/           # Raw video and coordinate data
├── models/            # Trained models and scalers
└── src/
    ├── preprocessing/ # Data preparation scripts
    ├── modeling/      # ML model training and prediction
    └── plotting/      # Visualization scripts
```

## Workflow

### 1. Data Preprocessing

#### Frame Extraction and Correction
**Script:** `src/preprocessing/frame_sampler.py`
- Extracts 100 evenly spaced frames from drone videos
- Applies fisheye correction using the `defisheye` library
- Outputs 2160x2160 images at 96 DPI for consistent conversion

#### Manual Position Labeling
**Script:** `src/preprocessing/labeling.py`
- Interactive tool for manually labeling drone positions
- Normalizes and saves coordinates to Excel
- Supports both left and right camera views

#### Data Alignment and Formatting
**Script:** `src/preprocessing/align_and_format.py`
- Aligns timestamps between fisheye and OptiTrack data
- Converts normalized coordinates to millimeters
- Outputs aligned coordinate pairs to `input_coordinates.xlsx`
- Successfully matched 96 coordinate pairs

### 2. Machine Learning Model

#### Model Training
**Script:** `src/modeling/train_model.py`
- Implements a Multi-Layer Perceptron (MLP) model
- Features:
  - RobustScaler for input/output normalization
  - Multiple dense layers with batch normalization
  - Dropout layers for regularization
  - L2 regularization and Huber loss for stability
- Achieved Mean Absolute Error (MAE) of ~11.1cm
- Saves trained model and scalers to `models/` directory

#### Coordinate Prediction
**Script:** `src/modeling/predict_and_export.py`
- Loads trained model and scalers
- Generates 3D coordinate predictions
- Saves results to `mlp_predictions.xlsx`
- Performance metrics:
  - Overall MAE: 10.6cm
  - X-coordinate MAE: 3.5cm
  - Y-coordinate MAE: 16.3cm
  - Z-coordinate MAE: ~12cm

### 3. Results Visualization
**Script:** `src/plotting/plot_results.py`
- Creates comprehensive visualizations:
  1. Scatter plots comparing predicted vs true coordinates
  2. 3D trajectory visualization with multiple viewing angles
  3. Error analysis over time
  4. Error distribution analysis
- All plots are saved to `data/plots/` directory

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess Data:**
```bash
python src/preprocessing/align_and_format.py
```

2. **Train Model:**
```bash
python src/modeling/train_model.py
```

3. **Generate Predictions:**
```bash
python src/modeling/predict_and_export.py
```

4. **Visualize Results:**
```bash
python src/plotting/plot_results.py
```

## Results

The machine learning model successfully converts fisheye camera coordinates to 3D OptiTrack coordinates with varying accuracy across dimensions:
- Best performance in X-axis (3.5cm MAE)
- Moderate performance in Z-axis (~12cm MAE)
- Most challenging in Y-axis (16.3cm MAE)

Detailed visualizations of model performance can be found in the `data/plots/` directory.

## Notes
- All coordinate measurements are in millimeters
- Model performance varies by axis due to camera positioning and perspective effects
- Multiple viewing angles of 3D trajectories are provided for comprehensive analysis