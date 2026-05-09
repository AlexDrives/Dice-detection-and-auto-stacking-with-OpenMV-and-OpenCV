import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(str(path))


def collect_images(image_dir: Path):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def reprojection_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    per_image = []
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        point_count = len(projected)
        per_image.append(float(error / np.sqrt(point_count)))
        total_error += error * error
        total_points += point_count
    rms = float(np.sqrt(total_error / total_points)) if total_points else 0.0
    return rms, per_image


def save_opencv_yaml(path: Path, camera_matrix, dist_coeffs, image_size, rms_error, args, image_results):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("inner_cols", int(args.inner_cols))
    fs.write("inner_rows", int(args.inner_rows))
    fs.write("square_size_mm", float(args.square_mm))
    fs.write("rms_reprojection_error_px", float(rms_error))
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.release()

    meta_path = path.with_suffix(".json")
    meta = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "inner_cols": int(args.inner_cols),
        "inner_rows": int(args.inner_rows),
        "square_size_mm": float(args.square_mm),
        "rms_reprojection_error_px": float(rms_error),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "images": image_results,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a camera from checkerboard images.")
    parser.add_argument("--image-dir", default="images", help="folder containing checkerboard images")
    parser.add_argument("--inner-cols", type=int, default=9, help="number of inner corners along x")
    parser.add_argument("--inner-rows", type=int, default=6, help="number of inner corners along y")
    parser.add_argument("--square-mm", type=float, default=5.0, help="checkerboard square size in mm")
    parser.add_argument("--out-dir", default="output", help="output directory")
    parser.add_argument("--annotate", action="store_true", help="save images with detected corners")
    parser.add_argument("--undistort-sample", action="store_true", help="save one undistorted sample image")
    parser.add_argument("--use-sb", action="store_true", help="use OpenCV findChessboardCornersSB detector")
    parser.add_argument("--zero-tangent", action="store_true", help="fix tangential distortion to zero")
    parser.add_argument("--fix-k3", action="store_true", help="fix the third radial distortion coefficient")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    annotated_dir = out_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern_size = (args.inner_cols, args.inner_rows)
    objp = np.zeros((args.inner_cols * args.inner_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.inner_cols, 0 : args.inner_rows].T.reshape(-1, 2)
    objp[:, :2] *= args.square_mm

    objpoints = []
    imgpoints = []
    image_size = None
    image_results = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_paths = collect_images(image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    for path in image_paths:
        img = imread(path)
        if img is None:
            image_results.append({"file": str(path), "found": False, "reason": "read_failed"})
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        if args.use_sb:
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                cv2.CALIB_CB_EXHAUSTIVE
                + cv2.CALIB_CB_ACCURACY
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            corners2 = corners.astype(np.float32) if found else None
        else:
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) if found else None
        if found:
            objpoints.append(objp.copy())
            imgpoints.append(corners2)
            if args.annotate:
                drawn = img.copy()
                cv2.drawChessboardCorners(drawn, pattern_size, corners2, found)
                imwrite(annotated_dir / path.name, drawn)
        image_results.append({"file": str(path), "found": bool(found)})

    if len(objpoints) < 8:
        raise SystemExit(
            f"Only {len(objpoints)} valid checkerboard images found. "
            "Use at least 8, preferably 15-25 varied poses."
        )

    calibration_flags = 0
    if args.zero_tangent:
        calibration_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_k3:
        calibration_flags |= cv2.CALIB_FIX_K3

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=calibration_flags
    )
    rms_error, per_image_errors = reprojection_errors(
        objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
    )
    for result, error in zip([r for r in image_results if r.get("found")], per_image_errors):
        result["reprojection_error_px"] = error

    yaml_path = out_dir / "camera_intrinsics.yaml"
    save_opencv_yaml(yaml_path, camera_matrix, dist_coeffs, image_size, rms_error, args, image_results)

    if args.undistort_sample:
        sample_path = next(path for path, result in zip(image_paths, image_results) if result.get("found"))
        sample = imread(sample_path)
        h, w = sample.shape[:2]
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(sample, camera_matrix, dist_coeffs, None, new_matrix)
        imwrite(out_dir / f"undistorted_{sample_path.name}", undistorted)
        x, y, rw, rh = roi
        image_results.append({"undistort_sample": str(sample_path), "roi": [int(x), int(y), int(rw), int(rh)]})

    print(f"Valid images: {len(objpoints)} / {len(image_paths)}")
    print(f"RMS reprojection error: {rms_error:.4f} px")
    print(f"OpenCV ret: {ret:.4f}")
    print(f"Saved: {yaml_path}")
    print(f"Saved: {yaml_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
