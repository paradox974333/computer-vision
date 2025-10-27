# Radial Distortion Estimation - README

## Overview

I took this code to implement camera calibration and radial distortion correction using the **Division Distortion Model**. This model estimates lens distortion by finding coefficients k1 and k2 that describe how light rays bend through the camera lens.

## Method Used

The implementation uses a **Division Distortion Model** with the following mathematical formulation:

```
x_d = x_u / (1 + k1*r² + k2*r⁴)
y_d = y_u / (1 + k1*r² + k2*r⁴)
```

where:
- `x_d, y_d` are distorted image coordinates
- `x_u, y_u` are undistorted image coordinates
- `r² = x_u² + y_u²` is the squared radial distance from the center
- `k1, k2` are radial distortion coefficients

The calibration process involves:
1. **Chessboard corner detection** using OpenCV
2. **RANSAC filtering** to remove outliers
3. **Non-linear optimization** using least-squares method to estimate camera intrinsics and distortion parameters
4. **Image undistortion** by inverse mapping

## Photos Used

- Multiple photos taken in my home setup with a chessboard calibration pattern
- One internet photo: `black-and-white-floor-tile-2016384638-sp3qmdtk.jpg`

## Input Parameters

- **image**: Input image containing a chessboard pattern (BGR or grayscale)
- **grid_size** (optional): Chessboard dimensions as (width, height) in number of inner corners
  - Default: Auto-detects from common patterns like (9,6), (6,9), (8,6), etc.
- **square_size** (optional): Physical size of chessboard squares in world units
  - Default: 1.0

## Output Parameters

The `optimize()` method returns a dictionary containing:

- **k1**: First radial distortion coefficient
- **k2**: Second radial distortion coefficient
- **K**: 3×3 camera intrinsic matrix containing focal lengths (fx, fy) and principal point (cx, cy)
- **R**: 3×3 rotation matrix (camera pose)
- **t**: 3×1 translation vector (camera position)
- **world_points**: 3D coordinates of detected chessboard corners
- **image_points**: 2D pixel coordinates of detected corners

Additional outputs:
- **undistorted_image.jpg**: Corrected image with distortion removed
- **Reprojection Error**: Average pixel error between detected and reprojected corners (printed to console)

## Usage

```python
estimator = RadialDistortionEstimator()
results = estimator.optimize(image)
undistorted_image = estimator.undistort_image(image)
```




##example input and output
