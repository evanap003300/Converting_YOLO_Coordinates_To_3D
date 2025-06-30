import os
import cv2
import pandas as pd

# Data for excel file
data_rows = []

# Create a folder to save the images with the marked points if it doesn't exist
output_folder = 'Marked_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Global variable to store the clicked point coordinates
clicked_point = None

# Function to add hand labeled data to excel file
def add_data(x, y):
    # Image Dimentions: Width: 2160, Height: 2160
    x_norm = x / 2160.0
    y_norm = y / 2160.0

    # x_3d just equals x_norm if going from right camera
    # x_3d = x_norm
    x_3d = 1 - x_norm
    y_3d = 1 - y_norm
    
    # Add data to the excel file
    data_rows.append({
        "x": x,
        "y": y,
        "x-norm": x_norm,
        "y-norm": y_norm,
        "x-3d": x_3d,
        "y-3d": y_3d
    })

# Function to be called when a point is clicked
def on_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
        clicked_point = (x, y)
        add_data(x, y)
        # Mark the clicked point with a red cross
        cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=6)
        
        # Print the X, Y coordinates on the image
        cv2.putText(image, f'({x},{y})', (x + 100, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
        
        # Show the image with the marked point
        cv2.imshow("Image", image)

# Folder where the images are located
image_folder = 'output_frames/defisheye/Right_Frames'

# List all PNG images in the folder
# List all PNG images in natural numeric order
image_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith('.png')],
    key=lambda x: int(''.join(filter(str.isdigit, x)))
)

# Loop through each image file
for image_name in image_files:
    # Read the image
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    
    # Show the image and set up the mouse callback for clicking
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", on_click)

    # Wait until the user clicks on the image
    while clicked_point is None:
        cv2.waitKey(1)  # Wait for a key event or mouse click
        
    print(image_name)

    # Save the image with the marked point (Commented out for now)
    # output_path = os.path.join(output_folder, image_name)
    # cv2.imwrite(output_path, image)

    # Reset clicked point for the next image
    clicked_point = None

    # Close the image window after processing
    cv2.destroyAllWindows()

# Make excel file 
df = pd.DataFrame(data_rows)

# Name of file to save to
df.to_excel("Right_Images_Data.xlsx", index=False)

print("Right image data saved to Right_Images_Data.xlsx")

