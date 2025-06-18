# Drone Tracking with Fisheye Correction and 3D Coordinate Estimation

These scripts processes drone video footage from two fisheye cameras. It extracts frames, applies distortion correction, manually labels drone positions, and then attempts to convert them into a 3D coordinate system aligned with OptiTrack measurements.

---

## Files

- `frame_sampler.py` — Extracts 100 evenly spaced frames from a video and applies fisheye correction.
- `labeling.py` — Tool for manually labeling drone positions in images.
- `convert.py` — Converts labeled 2D coordinates into 3D space using calibration, rotation, and transformation logic.

---

## Workflow

**1. Extract and Undistort Frames**

**Script:** `frame_sampler.py`

This script:
- Loads a trimmed drone video.
- Extracts 100 evenly spaced frames.
- Applies fisheye correction using the `defisheye` library.
- Saves output images at `2160x2160` with 96 DPI for consistent conversion.

**To Run:**
```bash
python frame_sampler.py
```

**2. Manually Label Drone Positions**

Script: labeling.py

This script:
	•	Opens each corrected image in order.
	•	Lets the user click on the drone’s position in each image.
	•	Normalizes and saves coordinates to Excel.

**To Run:**
```bash
python labeling.py
```

Update image_folder to match either the left or right camera image path before running.

**3. Convert to 3D OptiTrack-Aligned Coordinates**

Script: convert.py

This script:
	•	Reads in hand-labeled normalized coordinates from Excel.
	•	Converts coordinates from pixels to millimeters using DPI.
	•	Applies calibration corrections and transforms the data into 3D.
	•	Saves output as xyz_data.xlsx.

**To Run:**
```bash
python convert.py
```

Make sure to check that input_coordinates.xlsx exists in the correct location.

**Requirements:**
Install the following packages:
pip install -r requirements.txt

**Notes:**
	•	Assumes 2160x2160 images at 96 DPI.
	•	Z-axis derived from right camera X-pixel position.
	•	Camera tilt and position are modeled with a 20° rotation.
	•	Corrections and transformations can be further refined for accuracy.

**Next Steps**
	•	Improve scaling changes overtime.
	•	Explore training a small neural network to learn corrections weights.