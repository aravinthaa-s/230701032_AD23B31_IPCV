import cv2
import numpy as np
import os

from stereo import compute_disparity
from depth import compute_depth


# Paths
LEFT_FOLDER = "D:/ipcv_project/data/left"
RIGHT_FOLDER = "D:/ipcv_project/data/right"

OUTPUT_DISP = "D:/ipcv_project/output/disparity"
OUTPUT_DEPTH = "D:/ipcv_project/output/depth"

# Create output folders if not exist
os.makedirs(OUTPUT_DISP, exist_ok=True)
os.makedirs(OUTPUT_DEPTH, exist_ok=True)

# Get image list
image_files = sorted(os.listdir(LEFT_FOLDER))

for file in image_files:
    left_path = os.path.join(LEFT_FOLDER, file)
    right_path = os.path.join(RIGHT_FOLDER, file)

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    if left is None or right is None:
        print(f"Skipping {file} (not found)")
        continue

    print(f"Processing {file}...")

    # Step 1: Disparity
    disparity = compute_disparity(left, right)

    # Step 2: Depth
    depth = compute_depth(disparity)

    # Normalize disparity
    disp_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_display = disp_display.astype('uint8')

    # Normalize depth
    depth_display = np.clip(depth, 0, 100)
    depth_display = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = depth_display.astype('uint8')

    # Apply color maps
    disp_color = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    # Save outputs
    cv2.imwrite(os.path.join(OUTPUT_DISP, file), disp_color)
    cv2.imwrite(os.path.join(OUTPUT_DEPTH, file), depth_color)

    # Display (optional)
    cv2.imshow("Left", left)
    cv2.imshow("Disparity", disp_color)
    cv2.imshow("Depth", depth_color)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to stop
        break

cv2.destroyAllWindows()
print("✅ Processing completed!")