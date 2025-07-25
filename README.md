# Drone Tracking: 3D Coordinate Conversion from YOLO Predictions

This project processes drone position data derived from YOLO predictions and converts them into 3D OptiTrack-aligned coordinates. It employs a machine learning model for accurate coordinate conversion and provides comprehensive visualizations of model performance and error analysis.

## Project Scope & Organization

This repository contains all the code, data, and documentation necessary to reproduce the 3D coordinate conversion workflow and the figures presented in the associated research paper.

### Project Structure
```
3d-Coordinate-Conversions/
├── data/
│   ├── plots/          # Generated visualization outputs (PNG/JPG figures for the paper)
│   ├── predictions/    # Model predictions (e.g., mlp_predictions.xlsx)
│   ├── processed/      # Intermediate processed coordinate data (e.g., YOLO_Coordinates_mm.xlsx)
│   └── raw/            # Raw input data (e.g., YOLO_Coordinates.xlsx, OptiTrack_Data.xlsx)
├── models/             # Trained ML models and scalers
├── src/
│   ├── preprocessing/  # Data preparation scripts (e.g., align_and_format.py, now includes pixel-to-mm conversion)
│   ├── modeling/       # ML model training and prediction scripts (e.g., train_model.py, predict_and_export.py)
│   └── plotting/       # Visualization scripts (e.g., plot_results.py)
├── latex/              # LaTeX source files for the research paper
│   ├── figures/        # Final figure image files (PNG/JPG) referenced by the .tex document 
│   └── mlpuav-spie_v1.tex # Main LaTeX document for the paper
└── making_figures/     # Self-contained subfolders for reproducing each paper figure
├── Figure1_Drone_Setup/
├── Figure2_EPM_Closeup/
├── Figure3_3D_Trajectory/
├── Figure4_Error_Distribution/
├── Figure5_Scatter_Plots/
└── Figure6_Error_Over_Time/
└── ... (Each subfolder contains data and code for that specific figure)
```
## Workflow & Reproducibility

The workflow is designed to be highly reproducible. All code and data required for each step are provided.

### 1. Data Preprocessing

**Script:** `src/preprocessing/align_and_format.py`
- Loads raw YOLO coordinate data from `data/raw/YOLO_Coordinates.xlsx`.
- **Integrates the pixel-to-millimeter conversion logic to transform normalized pixel coordinates to physical dimensions.**
- Saves processed YOLO coordinates to `data/processed/YOLO_Coordinates_mm.xlsx`.

> **Note on `input_coordinates.xlsx`:** The combined input data for model training and prediction (`input_coordinates.xlsx`, located in `data/raw/` and within `making_figures` subfolders) is derived from `YOLO_Coordinates_mm.xlsx` and `OptiTrack_Data.xlsx`. Due to challenges with missing data points (e.g., drone leaving the frame), a specific subsection of the data was manually selected and curated to create `input_coordinates.xlsx`. This manual step is acknowledged, and the curated file is provided directly in relevant `making_figures` subfolders for immediate reproducibility of results. The raw source files are included for context.

### 2. Machine Learning Model

#### Model Training
**Script:** `src/modeling/train_model.py`
- Implements a Multi-Layer Perceptron (MLP) model.
- Employs RobustScaler for input/output normalization, dense layers with batch normalization, dropout for regularization, and L2 regularization with Huber loss for stability.
- Saves trained model and scalers to the `models/` directory.

#### Coordinate Prediction
**Script:** `src/modeling/predict_and_export.py`
- Loads the trained model and scalers from `models/`.
- Generates 3D coordinate predictions from the processed input data (`input_coordinates.xlsx`).
- Saves results to `data/predictions/mlp_predictions.xlsx`.
- Prints performance metrics (MAE, RMSE) for each coordinate.

### 3. Results Visualization & Figure Generation

**Script:** `src/plotting/plot_results.py`
- Creates comprehensive visualizations based on the model's predictions.
- Generates various plots including scatter comparisons, 3D trajectories, and error analyses.
- **Figures are styled for publication** (e.g., font sizes, colormaps, DPI]).
- All generated plots are saved to the `data/plots/` directory.
- **Final figures used in the LaTeX document are copied from `data/plots/` to `latex/figures/` for direct inclusion in the paper .**

#### Reproducing Specific Figures

Each subfolder within the `making_figures/` directory is designed to be self-contained. To reproduce a specific figure from the paper:
1.  Navigate to the corresponding `making_figures/FigureX_DescriptiveName/` subfolder.
2.  Review the `README.md` within that subfolder for any specific instructions related to that figure's generation or data sourcing.
3.  Execute the provided `generate_figure.sh` (or `generate_figure.py`) script, which will run the necessary plotting code and regenerate the figure image.

## Research Paper

The LaTeX source code for the research paper, which includes all the integrated figures, is located in the `latex/` folder. The main document is `latex/mlpuav-spie_v1.tex`.

## Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/ARTS-Laboratory/Paper-2026-MLPUAV-SPIE.git](https://github.com/ARTS-Laboratory/Paper-2026-MLPUAV-SPIE.git)
    cd Paper-2026-MLPUAV-SPIE
    ```
2.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install LaTeX distribution:**
    * **macOS:** Install [MacTeX](https://www.tug.org/mactex/) (includes TeX Live).
    * **Windows:** Install [MiKTeX](https://miktex.org/download).
    * **Linux:** Install [TeX Live](https://www.tug.org/texlive/acquire-iso.html).
4.  **Install LaTeX Editor (TeXstudio preferred):**
    * Download [TeXstudio](https://www.texstudio.org/).

## Usage

1.  **Prepare Data (Manual Step acknowledged):** Ensure `data/raw/YOLO_Coordinates.xlsx` and `data/raw/OptiTrack_Data.xlsx` are in place, and if curating data manually, place `input_coordinates.xlsx` in `data/raw/`.
2.  **Run Full Workflow (or individual steps):**
    ```bash
    python src/preprocessing/align_and_format.py
    python src/modeling/train_model.py
    python src/modeling/predict_and_export.py
    python src/plotting/plot_results.py
    ```
3.  **Compile LaTeX Document:** Open `latex/mlpuav-spie_v1.tex` in TeXstudio and compile (run BibTeX then compile twice for citations).
4.  **Reproduce Individual Figures:** See instructions under "Reproducing Specific Figures" above.

## Results

The machine learning model successfully converts YOLO data to 3D OptiTrack coordinates, demonstrating varying accuracy across dimensions. Detailed visualizations of model performance can be found in the `data/plots/` directory, and are integrated into the LaTeX paper.

## Notes
-   All coordinate measurements are in millimeters.
-   Raw video processing and manual labeling are not included in this repository; the workflow starts from provided YOLO and OptiTrack Excel files.
