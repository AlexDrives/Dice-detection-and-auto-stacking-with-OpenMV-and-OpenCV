# OpenMV Dice Stacking Pipeline

OpenMV Dice Stacking Pipeline is a PC-side vision and robot-command workflow for detecting dice captured by an OpenMV camera, mapping dice centers from pixels to world coordinates, and generating serial commands for a suction-based stacking robot.

The project keeps OpenMV responsible for image acquisition and moves the heavier work to the PC:

1. capture images from OpenMV
2. detect dice centers and values on the PC
3. convert pixel coordinates to world coordinates with calibration data
4. generate and optionally send robot stacking commands

## Demo

![Dice stacking demo](assets/6dice_stack_demo.gif)

The GIF above is compressed and sped up for README viewing. Open the original video for the full-resolution demo:

[Watch the dice stacking demo](assets/6dice_stack.mp4)

## Repository Contents

Core scripts:

- `pc_openmv_capture_once.py`: trigger an OpenMV capture run, then pull JPG files over the serial REPL.
- `pc_openmv_serial_pull.py`: pull an already captured image batch from OpenMV over serial.
- `pc_openmv_disk_capture.py`: copy captured images from the mounted OpenMV drive.
- `pc_dice_detect.py`: OpenCV-based dice detector for local capture images.
- `yolo_dice_infer.py`: optional Ultralytics YOLO inference path for dice value detection.
- `camera_undistort.py`: camera calibration loading and undistortion helpers.
- `validate_calibration_models.py`: compare and validate pixel-to-world calibration models.
- `dice_stack_pipeline.py`: end-to-end detection-output to stacking-command pipeline.
- `dice_stack_commands.py`: generate stack commands from world-coordinate input.
- `dice_restore_commands.py`: generate reverse/restore commands.

Configuration and calibration:

- `dice_stack_config.yaml`: command-generation parameters for the simpler stack/restore scripts.
- `dice_stack_pipeline_config.yaml`: full pipeline parameters.
- `calibrations/`: camera and pixel-to-world calibration files.
- `coords/semi_auto_calibration_points.csv`: curated calibration points used by validation tooling.
- `相机标定/`: helper scripts for checkerboard generation and OpenCV camera calibration.

Local outputs such as `captures/`, `results/`, `logs/`, raw calibration images, notebooks, videos, and extracted third-party app files are intentionally ignored by Git.

## Requirements

Python 3.10+ is recommended.

Install the base dependencies:

```powershell
pip install -r requirements.txt
```

`ultralytics` is only needed if you use `yolo_dice_infer.py`. For the OpenMV serial tools, make sure the board is visible as a serial port such as `COM17`.

## Quick Start

Capture a new image batch from OpenMV:

```powershell
python pc_openmv_capture_once.py --port COM17 --expected 6
```

The images are saved under:

```text
captures/openmv_YYYYMMDD_HHMMSS/
```

Run the OpenCV detector on one capture folder:

```powershell
python pc_dice_detect.py --input captures/openmv_YYYYMMDD_HHMMSS --body-threshold 180 --save-binary
```

If lighting changes, compare thresholds such as `170`, `175`, `180`, and `185`.

Generate stack commands from a world-coordinate CSV or JSON file:

```powershell
python dice_stack_commands.py --coords-file coords/example_world_coords.csv
```

Run the fuller pipeline with calibrated pixel-to-world conversion:

```powershell
python dice_stack_pipeline.py --input results/dice_detect/frame_01.json
```

Review `dice_stack_pipeline_config.yaml` before sending commands to hardware. Motion limits, pickup height, stack target, suction timing, and serial command names should match your robot controller.

## Camera Calibration

The `相机标定/` folder contains the calibration utilities:

```powershell
cd 相机标定
python generate_checkerboard.py
python calibrate_camera.py --annotate --undistort-sample
```

Generated camera intrinsics can be copied into `calibrations/` and passed to detection with:

```powershell
python pc_dice_detect.py --input captures/openmv_YYYYMMDD_HHMMSS --intrinsics calibrations/camera_intrinsics_paper_9p63_simple.json
```

## YOLO Path

For a trained dice model:

```powershell
python yolo_dice_infer.py --model best.pt --source 0 --show
```

Recommended class names are `d1`, `d2`, `d3`, `d4`, `d5`, and `d6`. The script maps class names to dice values and outputs per-frame detections with centers.

## Notes

- The project is tuned for a specific OpenMV camera, lighting setup, calibration board, and robot coordinate frame.
- Treat the included calibration files as examples or starting points. Recalibrate after changing camera position, lens, board height, or robot fixture.
- Test command output without `--send` before moving hardware.

## License

MIT License. See `LICENSE`.
