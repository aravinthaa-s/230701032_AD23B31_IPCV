import cv2
import numpy as np

def compute_disparity(left_img, right_img):
    """
    Compute disparity map using StereoSGBM
    """

    # Convert to grayscale
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # StereoSGBM parameters (tuned)
    min_disp = 0
    num_disp = 16 * 8  # must be divisible by 16
    block_size = 7

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,

        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,

        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=32
    )

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Noise reduction
    disparity = cv2.medianBlur(disparity, 5)

    return disparity