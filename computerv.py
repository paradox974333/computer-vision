import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Tuple


class RadialDistortionEstimator:
    """
    Division Distortion Model:
    x_d = x_u / (1 + k1*r^2 + k2*r^4)
    y_d = y_u / (1 + k1*r^2 + k2*r^4)
    where r^2 = x_u^2 + y_u^2, k1, k2 are distortion coefficients
    """
    
    def __init__(self):
        self.k1 = 0.0
        self.k2 = 0.0
        self.K = None
        self.R = None
        self.t = None
    
    def detect_corners(self, image: np.ndarray, grid_size: Tuple[int, int] = None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if grid_size is None:
            grid_sizes = [(9, 6), (6, 9), (8, 6), (6, 8), (7, 5), (5, 7), (10, 7), (7, 10)]
        else:
            grid_sizes = [grid_size, (grid_size[1], grid_size[0])]
        
        for gs in grid_sizes:
            ret, corners = cv2.findChessboardCorners(gray, gs, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return corners.reshape(-1, 2), gs
        
        for gs in grid_sizes[:4]:
            ret, corners = cv2.findChessboardCorners(gray, gs, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return corners.reshape(-1, 2), gs
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        for gs in grid_sizes:
            ret, corners = cv2.findChessboardCorners(enhanced, gs, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return corners.reshape(-1, 2), gs
        
        return None, None
    
    def create_world_points(self, grid_size: Tuple[int, int], square_size: float = 1.0):
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
        objp *= square_size
        return objp
    
    def apply_distortion(self, points: np.ndarray, k1: float, k2: float):
        r2 = np.clip(points[:, 0]**2 + points[:, 1]**2, 0, 1e10)
        factor = 1.0 / np.maximum(1.0 + k1 * r2 + k2 * r2**2, 1e-10)
        return points * factor[:, np.newaxis]
    
    def remove_distortion(self, points: np.ndarray, k1: float, k2: float, cx: float, cy: float, fx: float, fy: float):
        x_d = (points[:, 0] - cx) / fx
        y_d = (points[:, 1] - cy) / fy
        x_u, y_u = x_d.copy(), y_d.copy()
        
        for _ in range(10):
            r2 = np.clip(x_u**2 + y_u**2, 0, 1e10)
            factor = 1.0 / np.maximum(1.0 + k1 * r2 + k2 * r2**2, 1e-10)
            x_u = x_d / factor
            y_u = y_d / factor
        
        return np.column_stack([x_u * fx + cx, y_u * fy + cy])
    
    def project_points(self, world_points, K, R, t, k1, k2):
        points_cam = (R @ world_points.T).T + t.reshape(1, 3)
        z = np.maximum(points_cam[:, 2:3], 1e-10)
        points_norm = points_cam[:, :2] / z
        points_dist = self.apply_distortion(points_norm, k1, k2)
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        return np.column_stack([points_dist[:, 0] * fx + cx, points_dist[:, 1] * fy + cy])
    
    def cost_function(self, params, world_points, image_points, img_shape):
        fx, fy, cx, cy = params[:4]
        k1, k2 = params[4:6]
        rvec = params[6:9]
        tvec = params[9:12]
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R, _ = cv2.Rodrigues(rvec)
        
        projected = self.project_points(world_points, K, R, tvec, k1, k2)
        return (projected - image_points).ravel()
    
    def ransac_filter(self, world_points, image_points, threshold=5.0, iterations=1000):
        n_points = len(image_points)
        best_inliers = np.ones(n_points, dtype=bool)
        best_count = 0
        
        if n_points < 12:
            return best_inliers
        
        for i in range(iterations):
            idx = np.random.choice(n_points, min(12, n_points), replace=False)
            
            try:
                H, _ = cv2.findHomography(world_points[idx, :2], image_points[idx], 0)
                if H is None:
                    continue
                
                world_h = np.column_stack([world_points[:, :2], np.ones(n_points)])
                proj_h = (H @ world_h.T).T
                proj = proj_h[:, :2] / np.maximum(proj_h[:, 2:3], 1e-10)
                
                errors = np.linalg.norm(proj - image_points, axis=1)
                inliers = errors < threshold
                count = np.sum(inliers)
                
                if count > best_count:
                    best_count = count
                    best_inliers = inliers
            except:
                continue
        
        return best_inliers
    
    def initialize_params(self, image_points, world_points, img_shape):
        h, w = img_shape[:2]
        fx = fy = max(w, h)
        cx, cy = w / 2, h / 2
        
        K_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        success, rvec, tvec = cv2.solvePnP(world_points, image_points, K_init, None)
        
        if not success:
            rvec = np.zeros(3)
            tvec = np.array([0, 0, 5])
        
        return np.concatenate([[fx, fy, cx, cy], [0.0, 0.0], rvec.ravel(), tvec.ravel()])
    
    def optimize(self, image, grid_size=None, square_size=1.0):
        img_shape = image.shape
        
        image_points, detected_grid = self.detect_corners(image, grid_size)
        if image_points is None:
            raise ValueError("Could not detect checkerboard corners")
        
        world_points = self.create_world_points(detected_grid, square_size)
        
        inlier_mask = self.ransac_filter(world_points, image_points)
        world_points = world_points[inlier_mask]
        image_points = image_points[inlier_mask]
        
        if len(image_points) < 10:
            raise ValueError(f"Too few inliers for calibration")
        
        params_init = self.initialize_params(image_points, world_points, img_shape)
        
        h, w = img_shape[:2]
        bounds = (
            [-np.inf, -np.inf, 0, 0, -1.0, -1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, w, h, 1.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        )
        
        result = least_squares(self.cost_function, params_init, args=(world_points, image_points, img_shape),
                              method='trf', bounds=bounds, max_nfev=500, ftol=1e-8, xtol=1e-8, verbose=0)
        
        params = result.x
        fx, fy, cx, cy = params[:4]
        self.k1, self.k2 = params[4:6]
        
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.R, _ = cv2.Rodrigues(params[6:9])
        self.t = params[9:12]
        
        return {'k1': self.k1, 'k2': self.k2, 'K': self.K, 'R': self.R, 't': self.t,
                'world_points': world_points, 'image_points': image_points}
    
    def undistort_image(self, image):
        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        points = np.column_stack([x.ravel(), y.ravel()])
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        undist_points = self.remove_distortion(points, self.k1, self.k2, cx, cy, fx, fy)
        
        map_x = undist_points[:, 0].reshape(h, w).astype(np.float32)
        map_y = undist_points[:, 1].reshape(h, w).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    def compute_undistorted_grid(self, image_points):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return self.remove_distortion(image_points, self.k1, self.k2, cx, cy, fx, fy)
    
    def compute_reprojection_error(self, world_points, image_points):
        reprojected = self.project_points(world_points, self.K, self.R, self.t, self.k1, self.k2)
        residuals = reprojected - image_points
        errors = np.linalg.norm(residuals, axis=1)
        return np.mean(errors), residuals


def main():
    image = cv2.imread('black-and-white-floor-tile-2016384638-sp3qmdtk.jpg')
    estimator = RadialDistortionEstimator()
    
    results = estimator.optimize(image)
    undistorted_image = estimator.undistort_image(image)
    undistorted_grid = estimator.compute_undistorted_grid(results['image_points'])
    mean_error, residuals = estimator.compute_reprojection_error(results['world_points'], results['image_points'])
    
    cv2.imwrite('undistorted_image.jpg', undistorted_image)
    
    print(f"Distortion Model: Division Model")
    print(f"k1={results['k1']:.8f}, k2={results['k2']:.8f}")
    print(f"Reprojection Error: {mean_error:.4f} px")
    
    return estimator, results


if __name__ == "__main__":
    estimator, results = main()
