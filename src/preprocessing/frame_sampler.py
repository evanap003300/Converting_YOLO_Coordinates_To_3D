import cv2
import os
import numpy as np
from defisheye import Defisheye
from PIL import Image

# Parameters for fisheye correction
dtype = "equalarea"     # Type of fisheye projection
format = "fullframe"    # Format of the input fisheye image
fov = 160               # Field of view of the original camera lens in degrees
pfov = 90               # Desired field of view after correction
dpi = (96, 96)          # Target DPI for saved output images

# Applies fisheye correction to a single image and saves the output with updated DPI.
def defisheye_frame(input_path, output_path):
    obj = Defisheye(input_path, dtype=dtype, format=format, fov=fov, pfov=pfov)
    obj.convert(outfile=output_path)

    # Sets DPI
    with Image.open(output_path) as img:
        img.save(output_path, dpi=dpi)

# Extracts a set number of frames from the video, applies fisheye correction and saves the corrected frames with consistent DPI.
def extract_and_defisheye_frames(video_path, output_folder, num_frames=100, prefix="L"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Retrieve video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    # Output directory for corrected (defisheyed) frames
    fixed_folder = os.path.join(output_folder, "defisheye")
    os.makedirs(fixed_folder, exist_ok=True)

    # Generate evenly spaced frame indices across the entire video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipped unreadable frame at {frame_idx}")
            continue

        # Temporarily save the raw frame
        temp_path = os.path.join(fixed_folder, "temp_frame.png")
        cv2.imwrite(temp_path, frame)

        # Apply fisheye correction and save final output image
        output_path = os.path.join(fixed_folder, f"{prefix}{i+1:02d}_fixed.png")
        defisheye_frame(temp_path, output_path)
        print(f"Saved defisheyed frame: {output_path}")

        # Delete temporary raw frame file
        os.remove(temp_path)

    # Release video resource
    cap.release()
    print("Done: All frames extracted and defisheyed.")

# Example usage (adjust path and prefix as needed)
extract_and_defisheye_frames("../videos/left_trimmed_fixed.mp4", "output_frames", prefix="L")