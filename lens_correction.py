"""
Lens Correction Module for PPE Detection System

Provides fish-eye/lens distortion correction for images from three camera types:
- Left/Right: 3840x1920 equirectangular (360-degree cameras)
- Down: 2340x2160 circular fisheye (ceiling-mounted)

Usage:
    from lens_correction import LensCorrector

    corrector = LensCorrector()
    corrected_img = corrector.undistort(img, camera_id='left')
"""

import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


@dataclass
class CameraCalibration:
    """Stores calibration parameters for a single camera."""
    camera_id: str
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_size: Tuple[int, int]  # (width, height)
    is_fisheye: bool
    model_type: str  # 'equirectangular', 'circular_fisheye', 'standard'

    # Pre-computed undistortion maps (computed once, reused)
    map1: Optional[np.ndarray] = field(default=None, repr=False)
    map2: Optional[np.ndarray] = field(default=None, repr=False)


class LensCorrector:
    """
    Handles lens distortion correction for multiple cameras.

    Pre-computes undistortion maps for efficiency when processing many images.
    """

    def __init__(self, config_path: str = "camera_calibration.json"):
        """
        Initialize lens corrector with calibration configuration.

        Args:
            config_path: Path to JSON file with camera calibration parameters.
                        Creates default config if file doesn't exist.
        """
        self.config_path = Path(config_path)
        self.calibrations: Dict[str, CameraCalibration] = {}
        self.enabled = True
        self.balance = 0.0  # 0 = full crop, 1 = full image with black borders

        self._load_calibrations()

    def _load_calibrations(self) -> None:
        """Load calibration parameters from JSON config or use defaults."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            self.enabled = config.get('settings', {}).get('enabled', True)
            self.balance = config.get('settings', {}).get('balance', 0.0)

            for cam_id, cam_config in config.get('cameras', {}).items():
                K = np.array(cam_config['camera_matrix'], dtype=np.float64)
                D = np.array(cam_config['distortion_coeffs'], dtype=np.float64)

                calib = CameraCalibration(
                    camera_id=cam_id,
                    camera_matrix=K,
                    distortion_coeffs=D,
                    image_size=tuple(cam_config['image_size']),
                    is_fisheye=cam_config.get('is_fisheye', True),
                    model_type=cam_config.get('model_type', 'fisheye')
                )

                self.calibrations[cam_id] = calib
                self._compute_undistort_maps(calib)
        else:
            # Use default calibrations
            self._create_default_calibrations()

    def _create_default_calibrations(self) -> None:
        """Create default calibration parameters for the three cameras."""
        # Left camera: 3840x1920 equirectangular
        self.calibrations['left'] = CameraCalibration(
            camera_id='left',
            camera_matrix=np.array([
                [1920, 0, 1920],
                [0, 1920, 960],
                [0, 0, 1]
            ], dtype=np.float64),
            distortion_coeffs=np.array([-0.35, 0.12, 0.0, 0.0], dtype=np.float64),
            image_size=(3840, 1920),
            is_fisheye=True,
            model_type='equirectangular'
        )

        # Right camera: 3840x1920 equirectangular
        self.calibrations['right'] = CameraCalibration(
            camera_id='right',
            camera_matrix=np.array([
                [1920, 0, 1920],
                [0, 1920, 960],
                [0, 0, 1]
            ], dtype=np.float64),
            distortion_coeffs=np.array([-0.35, 0.12, 0.0, 0.0], dtype=np.float64),
            image_size=(3840, 1920),
            is_fisheye=True,
            model_type='equirectangular'
        )

        # Down camera: 2340x2160 circular fisheye
        self.calibrations['down'] = CameraCalibration(
            camera_id='down',
            camera_matrix=np.array([
                [1170, 0, 1170],
                [0, 1170, 1080],
                [0, 0, 1]
            ], dtype=np.float64),
            distortion_coeffs=np.array([-0.40, 0.15, 0.0, 0.0], dtype=np.float64),
            image_size=(2340, 2160),
            is_fisheye=True,
            model_type='circular_fisheye'
        )

        # Pre-compute maps for all cameras
        for calib in self.calibrations.values():
            self._compute_undistort_maps(calib)

    def _compute_undistort_maps(self, calib: CameraCalibration) -> None:
        """
        Pre-compute undistortion maps for efficient batch processing.

        Uses OpenCV fisheye model for fisheye cameras.
        """
        K = calib.camera_matrix
        D = calib.distortion_coeffs
        size = calib.image_size  # (width, height)

        if calib.is_fisheye:
            # Estimate new camera matrix with balance
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, size, np.eye(3), balance=self.balance
            )

            # Compute undistortion maps
            calib.map1, calib.map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, size, cv2.CV_16SC2
            )
        else:
            # Standard camera model
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 1, size)
            calib.map1, calib.map2 = cv2.initUndistortRectifyMap(
                K, D, None, new_K, size, cv2.CV_16SC2
            )

    def undistort(self, image: np.ndarray, camera_id: str) -> np.ndarray:
        """
        Apply distortion correction to an image.

        Args:
            image: Input BGR image as numpy array
            camera_id: Camera identifier ('left', 'right', 'down')

        Returns:
            Undistorted image. Returns original if camera_id not found or disabled.
        """
        if not self.enabled:
            return image

        # Normalize camera_id (extract from path-like strings)
        camera_id = self._normalize_camera_id(camera_id)

        if camera_id not in self.calibrations:
            return image

        calib = self.calibrations[camera_id]

        if calib.map1 is None or calib.map2 is None:
            return image

        # Apply undistortion using pre-computed maps
        undistorted = cv2.remap(
            image, calib.map1, calib.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        return undistorted

    def undistort_batch(self, images: List[np.ndarray], camera_id: str) -> List[np.ndarray]:
        """
        Apply distortion correction to multiple images from same camera.

        Args:
            images: List of input BGR images
            camera_id: Camera identifier

        Returns:
            List of undistorted images
        """
        return [self.undistort(img, camera_id) for img in images]

    def _normalize_camera_id(self, camera_id: str) -> str:
        """
        Normalize camera identifier from various input formats.

        Handles:
        - Direct IDs: 'left', 'right', 'down'
        - Path-like: '10-31_left', '11-7_down', 's3_images/10-31/left'
        """
        camera_id = camera_id.lower()

        # Direct match
        if camera_id in ['left', 'right', 'down']:
            return camera_id

        # Extract from path-like strings
        for cam in ['left', 'right', 'down']:
            if cam in camera_id:
                return cam

        return camera_id

    def get_camera_id_from_path(self, image_path: str) -> str:
        """
        Extract camera identifier from image file path.

        Parses paths like:
        - s3_images/10-31/left/image_xxx.jpg -> 'left'
        - s3_images/11-7/down/image_xxx.jpg -> 'down'
        - 10-31_left -> 'left'

        Args:
            image_path: Full path to image file

        Returns:
            Camera ID string ('left', 'right', or 'down')
        """
        path_lower = str(image_path).lower()

        # Check each camera type
        for cam in ['left', 'right', 'down']:
            if f'/{cam}/' in path_lower or f'_{cam}' in path_lower or f'\\{cam}\\' in path_lower:
                return cam

        return 'unknown'

    def is_enabled(self) -> bool:
        """Check if lens correction is enabled."""
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable lens correction."""
        self.enabled = enabled

    def get_calibration(self, camera_id: str) -> Optional[CameraCalibration]:
        """Get calibration parameters for a specific camera."""
        camera_id = self._normalize_camera_id(camera_id)
        return self.calibrations.get(camera_id)

    def save_config(self, path: Optional[str] = None) -> None:
        """Save current calibration to JSON file."""
        path = Path(path) if path else self.config_path

        config = {
            'version': '1.0',
            'settings': {
                'enabled': self.enabled,
                'balance': self.balance,
            },
            'cameras': {}
        }

        for cam_id, calib in self.calibrations.items():
            config['cameras'][cam_id] = {
                'camera_matrix': calib.camera_matrix.tolist(),
                'distortion_coeffs': calib.distortion_coeffs.tolist(),
                'image_size': list(calib.image_size),
                'is_fisheye': calib.is_fisheye,
                'model_type': calib.model_type,
            }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


def create_default_config(output_path: str = "camera_calibration.json") -> None:
    """
    Generate a template calibration config file with default parameters.

    Args:
        output_path: Where to save the config file
    """
    corrector = LensCorrector.__new__(LensCorrector)
    corrector.enabled = True
    corrector.balance = 0.0
    corrector.calibrations = {}
    corrector._create_default_calibrations()
    corrector.save_config(output_path)
    print(f"Created default config: {output_path}")


def visualize_correction(original_path: str,
                         corrector: LensCorrector,
                         output_path: Optional[str] = None) -> None:
    """
    Create a side-by-side comparison of original and corrected images.

    Args:
        original_path: Path to original image
        corrector: LensCorrector instance
        output_path: Where to save comparison (optional, displays if None)
    """
    img = cv2.imread(original_path)
    if img is None:
        print(f"Error: Could not read {original_path}")
        return

    camera_id = corrector.get_camera_id_from_path(original_path)
    corrected = corrector.undistort(img, camera_id)

    # Create side-by-side comparison
    h, w = img.shape[:2]
    scale = min(1200 / (w * 2), 600 / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_small = cv2.resize(img, (new_w, new_h))
    corrected_small = cv2.resize(corrected, (new_w, new_h))

    # Add labels
    cv2.putText(img_small, 'Original', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(corrected_small, f'Corrected ({camera_id})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    comparison = np.hstack([img_small, corrected_small])

    if output_path:
        cv2.imwrite(output_path, comparison)
        print(f"Saved comparison: {output_path}")
    else:
        cv2.imshow('Lens Correction Comparison', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Generate default config file
    create_default_config()

    # Test with sample image if available
    import sys
    if len(sys.argv) > 1:
        corrector = LensCorrector()
        visualize_correction(sys.argv[1], corrector)
