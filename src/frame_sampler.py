import cv2
import os
import numpy as np
from defisheye import Defisheye
from PIL import Image

# Parameters for defisheye
dtype = "equalarea"
format = "fullframe"
fov = 160
pfov = 90
dpi = (96, 96)  # target DPI

def defisheye_frame(input_path, output_path):
    obj = Defisheye(input_path, dtype=dtype, format=format, fov=fov, pfov=pfov)
    obj.convert(outfile=output_path)

    # Set DPI metadata
    with Image.open(output_path) as img:
        img.save(output_path, dpi=dpi)

def extract_and_defisheye_frames(video_path, output_folder, num_frames=100, prefix="L"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get frame info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"🎥 FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    # Output folder for defisheyed frames
    fixed_folder = os.path.join(output_folder, "defisheye")
    os.makedirs(fixed_folder, exist_ok=True)

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipped unreadable frame at {frame_idx}")
            continue

        # Save frame temporarily
        temp_path = os.path.join(fixed_folder, "temp_frame.png")
        cv2.imwrite(temp_path, frame)

        # Apply defisheye and overwrite with final output (with DPI)
        output_path = os.path.join(fixed_folder, f"{prefix}{i+1:02d}_fixed.png")
        defisheye_frame(temp_path, output_path)
        print(f"Saved defisheyed frame: {output_path}")

        # Clean temp file
        os.remove(temp_path)

    cap.release()
    print("Done: All frames extracted and defisheyed.")

# Example usage
extract_and_defisheye_frames("../videos/left_trimmed_fixed.mp4", "output_frames", prefix="L")