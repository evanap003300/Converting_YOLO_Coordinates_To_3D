# Drone Tracking: 3D Coordinate Conversion from YOLO Predictions

This project processes drone position data from YOLO predictions converts them into 3D coordinates aligned with OptiTrack measurements. It uses machine learning to achieve accurate coordinate conversion and provides comprehensive visualizations of model performance.

## Project Structure

```
3d-Coordinate-Conversions/
├── data/
│   ├── plots/          # Visualization outputs
│   ├── predictions/    # Model predictions
│   ├── processed/      # Processed coordinate data
│   └── raw/            # Raw coordinate data (YOLO, OptiTrack)
├── models/             # Trained models and scalers
└── src/
    ├── preprocessing/  # Data preparation scripts
    ├── modeling/       # ML model training and prediction
    └── plotting/       # Visualization scripts
```

## Workflow

### 1. Data Preprocessing

**Script:** `src/preprocessing/align_and_format.py`
- Loads YOLO coordinate data from `data/raw/YOLO_Coordinates.xlsx`
- Converts normalized coordinates to millimeters using camera calibration and transformation logic
- Saves processed coordinates to `data/processed/YOLO_Coordinates_mm.xlsx`
- (If needed, further alignment with OptiTrack data can be performed and saved as `input_coordinates.xlsx`)

> **Note:** This project assumes that YOLO and OptiTrack coordinate data are already available in Excel format. Frame extraction and manual labeling are not included in this repository.

### 2. Machine Learning Model

#### Model Training
**Script:** `src/modeling/train_model.py`
- Implements a Multi-Layer Perceptron (MLP) model
- Features:
  - RobustScaler for input/output normalization
  - Multiple dense layers with batch normalization
  - Dropout layers for regularization
  - L2 regularization and Huber loss for stability
- Saves trained model and scalers to `models/` directory

#### Coordinate Prediction
**Script:** `src/modeling/predict_and_export.py`
- Loads trained model and scalers
- Generates 3D coordinate predictions from processed input data
- Saves results to `data/predictions/mlp_predictions.xlsx`
- Prints performance metrics (MAE, RMSE) for each coordinate

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

The machine learning model successfully converts YOLO data to 3D OptiTrack coordinates with varying accuracy across dimensions.

Detailed visualizations of model performance can be found in the `data/plots/` directory.

## Notes
- All coordinate measurements are in millimeters
- Raw video processing and manual labeling are not included in this repository; the workflow starts from provided YOLO and OptiTrack Excel files.
