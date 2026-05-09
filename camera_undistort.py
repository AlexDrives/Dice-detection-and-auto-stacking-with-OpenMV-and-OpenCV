from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
        image_size = None
        if "image_width" in data and "image_height" in data:
            image_size = (int(data["image_width"]), int(data["image_height"]))
        return camera_matrix, dist_coeffs, image_size

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open camera calibration file: {path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise RuntimeError(f"Invalid camera calibration file: {path}")
    return camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64), (width, height)


def optimal_new_camera_matrix(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    alpha: float = 1.0,
) -> np.ndarray:
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha,
        image_size,
    )
    return new_matrix


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    new_matrix = optimal_new_camera_matrix(camera_matrix, dist_coeffs, (w, h), alpha)
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)
    return undistorted, new_matrix


def undistort_pixel_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    new_matrix = optimal_new_camera_matrix(camera_matrix, dist_coeffs, image_size, alpha)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=new_matrix)
    return undistorted.reshape(-1, 2), new_matrix
