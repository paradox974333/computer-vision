# Radial Distortion Estimation from Single Checkerboard Image

## What This Does

Ever notice how wide-angle camera photos make straight lines look curved? That's lens distortion, and this code figures out exactly how much your camera bends reality. Feed it a single photo of a checkerboard, and it'll calculate the distortion parameters, then straighten everything out.

I built this because I was curious whether I could calibrate a camera from just one image instead of the usual 10-20 photos most tutorials recommend. Turns out you can, if you're careful about it.

## The Math Behind It

Most camera calibration libraries use the **Brown-Conrady model** (what OpenCV calls), but I went with the **Division Distortion Model** instead. Why? It's actually more physically accurate for how lenses bend light, and the math is cleaner.

Here's the model:

```
x_distorted = x_undistorted / (1 + k1·r² + k2·r⁴)
y_distorted = y_undistorted / (1 + k1·r² + k2·r⁴)
```

where:
- **r² = x² + y²** is how far you are from the image center (squared)
- **k1, k2** are the distortion coefficients we're trying to find
- **x, y** are normalized coordinates (relative to focal length and principal point)

The division model is inverse-friendly: going from distorted to undistorted coordinates is just solving a polynomial, which converges fast with simple iteration.

### Why Division Over Brown-Conrady?

The standard model multiplies by distortion factors: `x_d = x_u(1 + k1·r² + k2·r⁴)`. Division models like mine are better for wide-angle lenses because:
- They naturally saturate (distortion can't go infinite)
- They model fisheye lenses more accurately
- Undistortion is numerically stable

## How It Works

### 1. Corner Detection (The Annoying Part)

Finding checkerboard corners sounds easy until you try it. The code tries multiple strategies because real photos are messy:

**First pass**: Try common grid sizes (9×6, 8×6, 7×5, etc.) with OpenCV's adaptive threshold detection.

**Second pass**: If that fails, use FAST_CHECK flag for speed but less accuracy.

**Last resort**: Apply CLAHE (histogram equalization) to enhance contrast. This saved me on several dimly-lit photos where the checker pattern was barely visible.

Once corners are found, I refine them to sub-pixel accuracy using `cornerSubPix` with an 11×11 window. This matters—moving corners by even 0.3 pixels changes your distortion estimate noticeably.

### 2. RANSAC Outlier Filtering

Here's something I learned the hard way: OpenCV's corner detection isn't perfect. Sometimes it'll confidently report corners that are completely wrong (especially near image edges or on partially occluded boards).

My RANSAC implementation:
- Randomly samples 12 corners and fits a homography
- Projects all corners through this homography
- Counts how many corners land within 5 pixels of detected positions
- Keeps the best consensus set after 1000 iterations

Why homography? Because the checkerboard is planar, so there exists a perfect projective transform between world and image coordinates (ignoring distortion). Bad corners violate this and get filtered out.

**Threshold choice**: 5 pixels was empirical. Too tight (2-3px) and you reject valid corners with slight detection noise. Too loose (10px+) and outliers sneak through.

### 3. The Optimization Problem

This is where the real work happens. I'm solving for 12 unknowns simultaneously:

- **fx, fy**: Focal lengths (can be different if pixels aren't square)
- **cx, cy**: Principal point (where optical axis hits the sensor)
- **k1, k2**: Distortion coefficients
- **rvec** (3 values): Rotation of checkerboard relative to camera (as Rodrigues vector)
- **tvec** (3 values): Translation of checkerboard

The cost function is simple: sum of squared reprojection errors. For each 3D world point:
1. Rotate and translate to camera coordinates
2. Project to normalized image plane (divide by Z)
3. Apply distortion model
4. Scale by focal length and shift by principal point
5. Compute distance to detected corner

I minimize this using scipy's `least_squares` with Trust Region Reflective algorithm (`method='trf'`). This handles bounds gracefully, which matters because:
- Principal point must be inside image boundaries
- Distortion coefficients typically range [-1, 1] (extreme distortion breaks everything)
- I use reasonable initial guesses (focal length ≈ image diagonal, principal point at center)

**Convergence**: Set ftol and xtol to 1e-8 for tight convergence. Max 500 iterations is usually overkill—most images converge in 50-100.

### 4. Image Undistortion

The tricky part about undistortion: you can't just transform pixel coordinates directly because you need to work backwards. For each pixel in the *output* image, you need to find where it came from in the *input* image.

My approach:
1. Generate a mesh grid of all output pixel coordinates
2. Convert to normalized camera coordinates (subtract principal point, divide by focal length)
3. Iteratively solve the inverse distortion equation:
   - Start with x_undistorted = x_distorted
   - Apply distortion to get estimated distorted position
   - Update x_undistorted based on error
   - Repeat 10 times (usually converges in 3-4)
4. Convert back to pixel coordinates
5. Use OpenCV's `remap` to sample the input image at these coordinates

This creates a smooth, interpolated undistorted image. Using `INTER_LINEAR` balances speed and quality—`INTER_CUBIC` is prettier but slower.

### 5. Reprojection Error

Finally, I measure how well the model fits by:
1. Taking the 3D world points of checkerboard corners
2. Projecting them through the estimated camera model (with distortion)
3. Computing pixel distance to the detected corners
4. Averaging across all corners

**Good calibration**: Mean reprojection error < 1 pixel
**Acceptable**: 1-2 pixels
**Something's wrong**: > 3 pixels (usually means bad corner detection or wrong grid size)

## What You Need

```python
import numpy as np
import cv2
from scipy.optimize import least_squares
```

That's it. No TensorFlow, no fancy ML. Just good old numerical optimization.

## Running It

```python
image = cv2.imread('your_checkerboard.jpg')
estimator = RadialDistortionEstimator()

# Auto-detect grid size and calibrate
results = estimator.optimize(image)

# Get the undistorted image
undistorted = estimator.undistort_image(image)

# Save results
cv2.imwrite('undistorted.jpg', undistorted)

# Check accuracy
print(f"k1 = {results['k1']:.8f}")
print(f"k2 = {results['k2']:.8f}")
print(f"Reprojection error = {mean_error:.4f} pixels")
```

Or if you know your grid size:
```python
results = estimator.optimize(image, grid_size=(9, 6))
```

## Results on Real Photos

### Example 1: Floor Tiles (Internet Photo)

**Input**: Black and white floor tiles with moderate barrel distortion

![black-and-white-floor-tile-2016384638-sp3qmdtk](https://github.com/user-attachments/assets/5b7ab45a-1fc0-4a9e-975c-f2f80fc53a3e)

**Output**: 
- k1 = -0.26424895, k2 = 0.07153628
- Reprojection error: 0.8721 pixels

<img width="723" height="70" alt="image" src="https://github.com/user-attachments/assets/6bf17cd5-b27e-4462-abbf-1f8a1c571ed0" />

![undistorted_image](https://github.com/user-attachments/assets/6ffcea15-0b37-4a61-854d-37b07d4f0985)

The negative k1 indicates barrel distortion (edges bow outward), which is typical for wide-angle lenses. The undistorted image shows perfectly straight tile lines.

### Example 2: Home Setup #1

**Input**: Printed checkerboard, slightly angled, decent lighting

![checkerboard_image3](https://github.com/user-attachments/assets/0886f1fe-3b4a-4e30-86c3-dfb979be23d1)

**Output**:
- k1 = -0.18533421, k2 = 0.03281847
- Reprojection error: 0.6354 pixels

<img width="690" height="58" alt="image" src="https://github.com/user-attachments/assets/7f441597-1b2c-47cd-a4ee-fbcbf266eab9" />

![undistorted_image](https://github.com/user-attachments/assets/4dbc3c68-a453-40b9-9c05-f53855a8fe89)

Less distortion than Example 1 (probably a phone camera at moderate zoom). Sub-pixel reprojection error means the model fits really well.

### Example 3: Home Setup #2

**Input**: Different angle, more oblique perspective

![checkerboard_image5](https://github.com/user-attachments/assets/6b3937d1-a0e4-4380-a658-ac1a84e3d10b)

**Output**:
- k1 = -0.21447836, k2 = 0.05129384
- Reprojection error: 0.7249 pixels

<img width="700" height="62" alt="image" src="https://github.com/user-attachments/assets/81e282bc-e288-4af5-b060-c4e61f67b519" />

![undistorted_image](https://github.com/user-attachments/assets/ca8cf6cb-d138-4cf6-859c-e79945ac5821)

Consistent distortion coefficients across photos from the same camera build confidence in the calibration.

## Things That Can Go Wrong

**"Could not detect checkerboard corners"**
- Print a higher-contrast checkerboard
- Ensure it fills 30-70% of the frame
- Add more lighting or reduce shadows
- Try rotating the board

**High reprojection error (>2 pixels)**
- Usually means the detected grid size is wrong
- Manually specify `grid_size=(width, height)` in inner corners
- Check if board is partially occluded
- Corners near image edges are less accurate

**Distorted image looks weird**
- Extreme distortion (|k1| > 0.5) needs more images for stable calibration
- Try adding bounds on k1, k2 to prevent unrealistic values
- Ensure principal point stays near image center

**Negative focal length or crazy intrinsics**
- Bug in initialization—shouldn't happen with current bounds
- Report it with the image

## Improvements I Considered But Didn't Do

- **Tangential distortion**: Didn't add p1, p2 terms because they're usually negligible (< 1% effect) for most lenses
- **Multi-image calibration**: Would improve accuracy but defeats the single-image challenge
- **Automatic grid size detection**: Could analyze detected corners to infer grid dimensions
- **Uncertainty quantification**: Could compute Jacobian at solution to estimate parameter covariance

## The Physics Intuition

Why does distortion happen? Real lenses aren't perfect. Light rays at the edge of the lens bend differently than rays through the center. This creates radial distortion:

- **Barrel distortion (k1 < 0)**: Wide-angle lenses, edges bow out
- **Pincushion distortion (k1 > 0)**: Telephoto lenses, edges bow in

The r² and r⁴ terms model increasing distortion as you move from center to edge. For most consumer cameras, k2 is small compared to k1—it's a higher-order correction.

