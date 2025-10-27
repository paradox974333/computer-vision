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




## example input and output
example 1:
input:
![black-and-white-floor-tile-2016384638-sp3qmdtk](https://github.com/user-attachments/assets/5b7ab45a-1fc0-4a9e-975c-f2f80fc53a3e)
output:
<img width="723" height="70" alt="image" src="https://github.com/user-attachments/assets/6bf17cd5-b27e-4462-abbf-1f8a1c571ed0" />
![undistorted_image](https://github.com/user-attachments/assets/6ffcea15-0b37-4a61-854d-37b07d4f0985)

eaxmple 2:
input:
![checkerboard_image3](https://github.com/user-attachments/assets/0886f1fe-3b4a-4e30-86c3-dfb979be23d1)
output:
<img width="690" height="58" alt="image" src="https://github.com/user-attachments/assets/7f441597-1b2c-47cd-a4ee-fbcbf266eab9" />
![undistorted_image](https://github.com/user-attachments/assets/4dbc3c68-a453-40b9-9c05-f53855a8fe89)

example 3:
input:
![checkerboard_image5](https://github.com/user-attachments/assets/6b3937d1-a0e4-4380-a658-ac1a84e3d10b)
output:
<img width="700" height="62" alt="image" src="https://github.com/user-attachments/assets/81e282bc-e288-4af5-b060-c4e61f67b519" />
![undistorted_image](https://github.com/user-attachments/assets/ca8cf6cb-d138-4cf6-859c-e79945ac5821)





