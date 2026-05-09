PC-side OpenMV pipeline

Goal:
- OpenMV only acquires images
- PC handles dice detection, center extraction, and later value recognition
- Then pixel-to-world calibration maps centers to world coordinates
- Existing stack command generator remains the final stage

Current working capture path:
- `E:\main.py` captures a fixed small batch on the board
- the PC pulls those JPGs back over the serial REPL
- each run only captures 6 images and overwrites the previous device-side batch

Environment:
- `conda activate image-processing`
- key packages:
  - `opencv-python`
  - `numpy`
  - optional later:
    - `ultralytics`

One-command capture:
- `python pc_openmv_capture_once.py --port COM17`

What that command does:
- interrupts to the OpenMV REPL
- re-runs `main.py`
- captures 6 frames into `E:\pc_frames`
- pulls those JPGs over serial
- saves them into:
  - `captures/openmv_YYYYMMDD_HHMMSS`

Helper scripts:
1. `pc_openmv_capture_once.py`
   Trigger capture and pull images in one step.
2. `pc_openmv_serial_pull.py`
   Pull an existing device-side JPG batch from `pc_frames`.
3. `pc_openmv_disk_capture.py`
   Copy from the mounted `E:` drive if Windows has refreshed the filesystem view.
4. `yolo_dice_infer.py`
   Local PC-side detector placeholder for the next stage.

Current OpenCV detector:
- `pc_dice_detect.py`
- Fixed dice-body binarization threshold defaults to `180`
- Run with binary mask output:
  - `python pc_dice_detect.py --input captures/openmv_YYYYMMDD_HHMMSS --save-dir results/dice_detect_test --body-threshold 180 --save-binary`
- If lighting changes, try `--body-threshold 170`, `175`, `180`, or `185` and compare the saved `*_binary_t*.png` image.

Recommended chain:
1. `pc_openmv_capture_once.py`
   Acquire a fresh small batch from OpenMV.
2. PC-side detector
   Run OpenCV or YOLO on the saved JPGs.
3. Calibration
   Fit pixel center -> world XY.
4. Command generation
   Feed world XY into `dice_stack_commands.py`.
