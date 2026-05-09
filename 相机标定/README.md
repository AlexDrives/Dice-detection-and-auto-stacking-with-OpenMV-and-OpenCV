# OpenCV Camera Calibration

This folder contains helper tools for calibrating the OpenMV camera with a printed checkerboard.

Recommended workflow:

1. generate or print a checkerboard
2. capture 15 to 25 calibration images from different positions and angles
3. run OpenCV calibration
4. copy the generated intrinsics into the repository-level `calibrations/` folder
5. use intrinsics before pixel-to-world homography when needed

## Files

- `generate_checkerboard.py`: generate printable checkerboard SVG/PNG assets.
- `calibrate_camera.py`: compute camera intrinsics and distortion coefficients from checkerboard images.
- `boards/`: generated checkerboard assets that are safe to keep in Git.
- `images/`: raw local calibration images, ignored by Git.
- `output*/`: generated calibration outputs and annotated debug images, ignored by Git.

## Generate A Checkerboard

Default output uses a 9 x 6 inner-corner board with 5 mm squares:

```powershell
python generate_checkerboard.py
```

Useful variants:

```powershell
python generate_checkerboard.py --inner-cols 9 --inner-rows 6 --square-mm 10
python generate_checkerboard.py --inner-cols 10 --inner-rows 7 --square-mm 5
```

Print at 100% scale. Disable "fit to page" or other printer scaling.

## Capture Images

Place calibration photos in `images/`. A good dataset usually includes:

- the board near the center and near all four corners
- slight tilt and rotation
- several distances
- sharp images with the full board visible
- no strong reflections or overexposure

## Run Calibration

```powershell
python calibrate_camera.py --annotate --undistort-sample
```

Expected outputs:

- `output/camera_intrinsics.yaml`
- `output/camera_intrinsics.json`
- `output/annotated/`
- `output/undistorted_*.jpg`

Quality rule of thumb:

- `< 0.3 px` RMS reprojection error: excellent
- `0.3 - 0.8 px`: usually usable
- `> 1.0 px`: recapture or inspect corner detection quality
