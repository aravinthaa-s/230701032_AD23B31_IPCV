import numpy as np

# KITTI camera parameters
FOCAL_LENGTH = 721.5
BASELINE = 0.54

def compute_depth(disparity):
    """
    Convert disparity map to depth map
    """

    # Avoid division by zero
    disparity = np.where(disparity <= 0, 0.1, disparity)

    # Depth formula
    depth = (FOCAL_LENGTH * BASELINE) / disparity

    return depth