from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "dice_stack_pipeline_config.yaml"
DEFAULT_RESULTS_DIR = ROOT / "results" / "stack_pipeline"
DEFAULT_LOG_DIR = ROOT / "logs" / "serial"


@dataclass(frozen=True)
class StackPipelineConfig:
    calibration: Path
    use_quadratic_correction: bool
    pickup_z: float
    dice_height: float
    target_x: float
    target_y: float
    target_z: float
    safe_margin_z: float
    move_time: float
    vertical_move_time: float
    release_lift_time: float
    speed: int
    recognition_x: float
    recognition_y: float
    recognition_z: float
    recognition_time: float
    move_to_recognition_at_start: bool
    move_to_recognition_at_end: bool
    send_delay_seconds: float
    suction_delay_seconds: float
    suction_repeat: int
    suction_settle_seconds: float
    emit_speed_command: bool
    start_with_suction_off: bool
    end_with_suction_off: bool
    lift_after_release: bool
    point_command: str
    line_command: str
    suction_command: str
    speed_command: str
    wait_command: str


@dataclass
class DiceItem:
    index: int
    value: int
    color: str
    pixel_x: float | None
    pixel_y: float | None
    world_x: float
    world_y: float
    raw: dict[str, object]


def format_number(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def command(prefix: str, *values: float) -> str:
    return f"{prefix}_" + ",".join(format_number(value) for value in values) + ","


def load_config(path: Path) -> StackPipelineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    target = raw.get("target") or {}
    recognition_pose = raw.get("recognition_pose") or {}
    calibration = Path(raw.get("calibration", "calibrations/dice_homography_20pt_quadratic_corrected.json"))
    if not calibration.is_absolute():
        calibration = ROOT / calibration
    return StackPipelineConfig(
        calibration=calibration,
        use_quadratic_correction=bool(raw.get("use_quadratic_correction", False)),
        pickup_z=float(raw.get("pickup_z", 79.0)),
        dice_height=float(raw.get("dice_height", 16.5)),
        target_x=float(target.get("x", 201.0)),
        target_y=float(target.get("y", 34.0)),
        target_z=float(target.get("z", raw.get("pickup_z", 79.0))),
        safe_margin_z=float(raw.get("safe_margin_z", 30.0)),
        move_time=float(raw.get("move_time", 360)),
        vertical_move_time=float(raw.get("vertical_move_time", raw.get("move_time", 360))),
        release_lift_time=float(raw.get("release_lift_time", raw.get("vertical_move_time", 520))),
        speed=int(float(raw.get("speed", 50))),
        recognition_x=float(recognition_pose.get("x", 301.6)),
        recognition_y=float(recognition_pose.get("y", 0.0)),
        recognition_z=float(recognition_pose.get("z", 304.2)),
        recognition_time=float(recognition_pose.get("time", 100)),
        move_to_recognition_at_start=bool(raw.get("move_to_recognition_at_start", True)),
        move_to_recognition_at_end=bool(raw.get("move_to_recognition_at_end", True)),
        send_delay_seconds=float(raw.get("send_delay_seconds", 0.15)),
        suction_delay_seconds=float(raw.get("suction_delay_seconds", 0.2)),
        suction_repeat=max(1, int(float(raw.get("suction_repeat", 1)))),
        suction_settle_seconds=float(raw.get("suction_settle_seconds", 0.0)),
        emit_speed_command=bool(raw.get("emit_speed_command", True)),
        start_with_suction_off=bool(raw.get("start_with_suction_off", True)),
        end_with_suction_off=bool(raw.get("end_with_suction_off", True)),
        lift_after_release=bool(raw.get("lift_after_release", True)),
        point_command=str(raw.get("point_command", "DescartesPoint")),
        line_command=str(raw.get("line_command", "DescartesLine")),
        suction_command=str(raw.get("suction_command", "Suction")),
        speed_command=str(raw.get("speed_command", "Speed")),
        wait_command=str(raw.get("wait_command", "Wait")),
    )


def load_calibration(path: Path, use_quadratic_correction: bool = True) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    homography = np.array(data["homography_pixel_to_world"], dtype=np.float64)
    if not use_quadratic_correction:
        return homography, np.zeros((6, 2), dtype=np.float64)
    correction = data.get("residual_correction") or {}
    coeffs = np.array(correction.get("coefficients_for_dx_dy", []), dtype=np.float64)
    if coeffs.shape != (6, 2):
        raise ValueError(f"Unsupported residual correction in {path}: expected 6x2 coefficients.")
    return homography, coeffs


def apply_homography(homography: np.ndarray, pixel_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(pixel_xy, dtype=np.float64).reshape(-1, 2)
    homo = np.c_[points, np.ones(len(points))]
    out = (homography @ homo.T).T
    return out[:, :2] / out[:, 2:3]


def apply_quadratic_correction(world_xy: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    x = world_xy[:, 0]
    y = world_xy[:, 1]
    features = np.c_[np.ones(len(world_xy)), x, y, x * x, x * y, y * y]
    return world_xy + features @ coeffs


def pixel_to_world(pixel_x: float, pixel_y: float, homography: np.ndarray, coeffs: np.ndarray) -> tuple[float, float]:
    raw_world = apply_homography(homography, np.array([[pixel_x, pixel_y]], dtype=np.float64))
    corrected = apply_quadratic_correction(raw_world, coeffs)
    return float(corrected[0, 0]), float(corrected[0, 1])


def read_json_or_csv(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return [dict(item) for item in data]

    rows = list(csv.DictReader(text.splitlines()))
    if rows:
        return [{key.strip(): value for key, value in row.items()} for row in rows]

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 2:
            continue
        items.append({"world_x": parts[0], "world_y": parts[1]})
    return items


def dice_from_records(records: list[dict[str, object]], cfg: StackPipelineConfig) -> list[DiceItem]:
    homography, coeffs = load_calibration(cfg.calibration, cfg.use_quadratic_correction)
    dice: list[DiceItem] = []
    for index, raw in enumerate(records):
        value = int(float(raw.get("value", index + 1)))
        if value <= 0:
            continue
        color = str(raw.get("color", "u"))

        if "world_x" in raw and "world_y" in raw:
            world_x = float(raw["world_x"])
            world_y = float(raw["world_y"])
            pixel_x = float(raw["pip_center_x"]) if "pip_center_x" in raw else None
            pixel_y = float(raw["pip_center_y"]) if "pip_center_y" in raw else None
        elif "x" in raw and "y" in raw and "pip_center_x" not in raw:
            world_x = float(raw["x"])
            world_y = float(raw["y"])
            pixel_x = None
            pixel_y = None
        else:
            pixel_x = float(raw.get("pip_center_x", raw.get("x")))
            pixel_y = float(raw.get("pip_center_y", raw.get("y")))
            world_x, world_y = pixel_to_world(pixel_x, pixel_y, homography, coeffs)

        dice.append(
            DiceItem(
                index=index,
                value=value,
                color=color,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                world_x=world_x,
                world_y=world_y,
                raw=raw,
            )
        )
    return dice


def sort_for_stack(dice: list[DiceItem]) -> list[DiceItem]:
    return sorted(dice, key=lambda item: (item.value, item.index))


def point_move(cfg: StackPipelineConfig, x: float, y: float, z: float, duration: float) -> str:
    return command(cfg.point_command, x, y, z, duration)


def line_move(cfg: StackPipelineConfig, x: float, y: float, z: float, duration: float) -> str:
    return command(cfg.line_command, x, y, z, duration)


def suction(cfg: StackPipelineConfig, enabled: bool) -> str:
    return command(cfg.suction_command, 1 if enabled else 0)


def wait(cfg: StackPipelineConfig, seconds: float) -> str:
    return command(cfg.wait_command, seconds)


def suction_sequence(cfg: StackPipelineConfig, enabled: bool) -> list[str]:
    commands = [suction(cfg, enabled) for _ in range(cfg.suction_repeat)]
    if cfg.suction_settle_seconds > 0:
        commands.append(wait(cfg, cfg.suction_settle_seconds))
    return commands


def speed(cfg: StackPipelineConfig) -> str:
    return command(cfg.speed_command, cfg.speed)


def generate_stack_commands(dice: Iterable[DiceItem], cfg: StackPipelineConfig) -> list[str]:
    commands: list[str] = []
    if cfg.emit_speed_command:
        commands.append(speed(cfg))
    if cfg.move_to_recognition_at_start:
        commands.append(
            point_move(
                cfg,
                cfg.recognition_x,
                cfg.recognition_y,
                cfg.recognition_z,
                cfg.recognition_time,
            )
        )
    if cfg.start_with_suction_off:
        commands.extend(suction_sequence(cfg, False))

    for stack_index, item in enumerate(dice):
        place_z = cfg.target_z + stack_index * cfg.dice_height
        safe_z = max(cfg.pickup_z, place_z) + cfg.safe_margin_z
        next_safe_z = max(cfg.pickup_z, place_z + cfg.dice_height) + cfg.safe_margin_z

        # Horizontal/approach move at safe height can be point motion. Every pure Z move is line motion.
        commands.extend(
            suction_sequence(cfg, False)
            + [
                point_move(cfg, item.world_x, item.world_y, safe_z, cfg.move_time),
                line_move(cfg, item.world_x, item.world_y, cfg.pickup_z, cfg.vertical_move_time),
            ]
            + suction_sequence(cfg, True)
            + [
                line_move(cfg, item.world_x, item.world_y, safe_z, cfg.vertical_move_time),
                point_move(cfg, cfg.target_x, cfg.target_y, safe_z, cfg.move_time),
                line_move(cfg, cfg.target_x, cfg.target_y, place_z, cfg.vertical_move_time),
            ]
            + suction_sequence(cfg, False)
        )
        if cfg.lift_after_release:
            commands.append(line_move(cfg, cfg.target_x, cfg.target_y, next_safe_z, cfg.release_lift_time))

    if cfg.end_with_suction_off:
        commands.extend(suction_sequence(cfg, False))
    if cfg.move_to_recognition_at_end:
        commands.append(
            point_move(
                cfg,
                cfg.recognition_x,
                cfg.recognition_y,
                cfg.recognition_z,
                cfg.recognition_time,
            )
        )
    return commands


def build_output(raw_records: list[dict[str, object]], dice: list[DiceItem], sorted_dice: list[DiceItem], commands: list[str], cfg: StackPipelineConfig) -> dict[str, object]:
    order_lookup = {id(item): order for order, item in enumerate(sorted_dice)}
    converted = []
    for item in dice:
        converted.append(
            {
                "input_index": item.index,
                "stack_order": order_lookup.get(id(item)),
                "value": item.value,
                "color": item.color,
                "pixel_x": None if item.pixel_x is None else round(item.pixel_x, 3),
                "pixel_y": None if item.pixel_y is None else round(item.pixel_y, 3),
                "world_x": round(item.world_x, 3),
                "world_y": round(item.world_y, 3),
                "raw": item.raw,
            }
        )

    stack_plan = []
    for stack_index, item in enumerate(sorted_dice):
        place_z = cfg.target_z + stack_index * cfg.dice_height
        safe_z = max(cfg.pickup_z, place_z) + cfg.safe_margin_z
        stack_plan.append(
            {
                "stack_order": stack_index,
                "input_index": item.index,
                "value": item.value,
                "color": item.color,
                "pickup_world": [round(item.world_x, 3), round(item.world_y, 3), round(cfg.pickup_z, 3)],
                "target_world": [round(cfg.target_x, 3), round(cfg.target_y, 3), round(place_z, 3)],
                "safe_z": round(safe_z, 3),
            }
        )

    return {
        "config": {
            "calibration": str(cfg.calibration),
            "pickup_z": cfg.pickup_z,
            "dice_height": cfg.dice_height,
            "target": {"x": cfg.target_x, "y": cfg.target_y, "z": cfg.target_z},
            "safe_margin_z": cfg.safe_margin_z,
            "recognition_pose": {
                "x": cfg.recognition_x,
                "y": cfg.recognition_y,
                "z": cfg.recognition_z,
                "time": cfg.recognition_time,
            },
            "move_time": cfg.move_time,
            "vertical_move_time": cfg.vertical_move_time,
            "line_command": cfg.line_command,
            "point_command": cfg.point_command,
            "suction_repeat": cfg.suction_repeat,
            "suction_settle_seconds": cfg.suction_settle_seconds,
        },
        "raw_detections": raw_records,
        "converted_dice": converted,
        "stack_plan": stack_plan,
        "commands": commands,
    }


def send_commands(commands: Iterable[str], port: str, baudrate: int, cfg: StackPipelineConfig) -> None:
    try:
        import serial
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: pyserial. Install with `pip install pyserial`.") from exc

    with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as ser:
        for command_text in commands:
            print(command_text)
            if command_text.startswith(f"{cfg.wait_command}_"):
                try:
                    delay = float(command_text.split("_", 1)[1].split(",", 1)[0])
                except ValueError:
                    delay = cfg.send_delay_seconds
            else:
                ser.write(f"{command_text}\r\n".encode("utf-8"))
                delay = cfg.suction_delay_seconds if command_text.startswith(f"{cfg.suction_command}_") else cfg.send_delay_seconds
            time.sleep(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert dice detections to world coordinates and generate serial commands for stacking any number of dice."
    )
    parser.add_argument("--input", type=Path, required=True, help="Detection JSON/CSV. pc_dice_detect frame_01.json is supported.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"Config YAML. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--output-json", type=Path, help="Detailed output JSON path.")
    parser.add_argument("--commands-out", type=Path, help="Plain text command output path.")
    parser.add_argument("--send", action="store_true", help="Send commands to serial after generating outputs.")
    parser.add_argument("--port", help="Serial port for --send, for example COM15.")
    parser.add_argument("--baudrate", type=int, default=115200)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_records = read_json_or_csv(args.input)
    dice = dice_from_records(raw_records, cfg)
    if not dice:
        raise SystemExit("No valid dice records found.")

    sorted_dice = sort_for_stack(dice)
    commands = generate_stack_commands(sorted_dice, cfg)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_json = args.output_json or DEFAULT_RESULTS_DIR / f"stack_pipeline_{timestamp}.json"
    commands_out = args.commands_out or DEFAULT_LOG_DIR / f"stack_commands_{timestamp}.txt"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    commands_out.parent.mkdir(parents=True, exist_ok=True)

    output = build_output(raw_records, dice, sorted_dice, commands, cfg)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    commands_out.write_text("\n".join(commands) + "\n", encoding="utf-8")

    print("Stack order:")
    for item in sorted_dice:
        print(
            "  value={value} input={index} world=({x},{y}) pixel=({px},{py})".format(
                value=item.value,
                index=item.index,
                x=format_number(round(item.world_x, 3)),
                y=format_number(round(item.world_y, 3)),
                px="-" if item.pixel_x is None else format_number(round(item.pixel_x, 3)),
                py="-" if item.pixel_y is None else format_number(round(item.pixel_y, 3)),
            )
        )

    print("\nCommands:")
    print("\n".join(commands))
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved commands: {commands_out}")

    if args.send:
        if not args.port:
            parser.error("--send requires --port")
        send_commands(commands, args.port, args.baudrate, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
