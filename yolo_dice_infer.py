from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install with `pip install ultralytics opencv-python`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PC-side YOLO dice inference. Outputs per-frame detections in an OBS-like format."
    )
    parser.add_argument("--model", type=Path, required=True, help="YOLO model path, e.g. best.pt")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index like 0, or a video/image path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--save-json", type=Path, help="Optional JSONL output path.")
    parser.add_argument("--show", action="store_true", help="Show annotated window.")
    return parser.parse_args()


def open_source(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def class_to_value(name: str) -> int | None:
    lower = name.lower().strip()
    mapping = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "dice1": 1,
        "dice2": 2,
        "dice3": 3,
        "dice4": 4,
        "dice5": 5,
        "dice6": 6,
        "d1": 1,
        "d2": 2,
        "d3": 3,
        "d4": 4,
        "d5": 5,
        "d6": 6,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    return mapping.get(lower)


def annotate_frame(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        value = det["value"]
        cx, cy = det["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        cv2.putText(
            frame,
            f"V{value}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    model = YOLO(str(args.model))
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    frame_id = 0
    save_file = None
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        save_file = args.save_json.open("w", encoding="utf-8")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            start = time.time()

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    name = str(names.get(cls_id, cls_id))
                    value = class_to_value(name)
                    if value is None:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append(
                        {
                            "value": value,
                            "center": (cx, cy),
                            "bbox": (x1, y1, x2, y2),
                            "conf": float(box.conf.item()),
                            "class_name": name,
                        }
                    )

            detections.sort(key=lambda d: (d["value"], d["center"][1], d["center"][0]))
            fps = 1.0 / max(1e-6, time.time() - start)

            obs_parts = []
            for det in detections:
                obs_parts.append(f'{det["center"][0]},{det["center"][1]},{det["value"]}')
            obs = "|".join(obs_parts)
            print(f"OBS:{frame_id},{len(detections)},{obs};FPS:{fps:.1f}")

            if save_file is not None:
                save_file.write(
                    json.dumps(
                        {
                            "frame_id": frame_id,
                            "fps": fps,
                            "detections": detections,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                save_file.flush()

            if args.show:
                annotated = frame.copy()
                annotate_frame(annotated, detections)
                cv2.imshow("YOLO Dice", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        if save_file is not None:
            save_file.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
