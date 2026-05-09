PC-side YOLO switch notes

Why:
- OpenMV-side classical vision has become fragile under lighting and blur changes.
- YOLO should run on the PC, not on the OpenMV board.

Current status:
- Added `yolo_dice_infer.py`
- It expects an Ultralytics YOLO model such as `best.pt`
- It outputs:
  `OBS:frame_id,count,x,y,value|x,y,value;FPS:f`

Recommended class names:
- `d1`, `d2`, `d3`, `d4`, `d5`, `d6`

Install on PC:
- `pip install ultralytics opencv-python`

Run examples:
- Camera 0:
  `python yolo_dice_infer.py --model best.pt --source 0 --show`
- Video file:
  `python yolo_dice_infer.py --model best.pt --source demo.mp4 --show`
- Save JSONL:
  `python yolo_dice_infer.py --model best.pt --source 0 --save-json logs/yolo_obs.jsonl`

Training suggestion:
- Train one detection class per dice value: `d1..d6`
- Use bbox center as the pickup pixel coordinate first
- If later you need more accurate pickup centers, switch to pose/keypoint labels
