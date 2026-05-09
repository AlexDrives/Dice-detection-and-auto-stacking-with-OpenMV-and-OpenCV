from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from exc


CONFIG_PATH = Path(__file__).with_name("dice_stack_config.yaml")


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class StackConfig:
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
    release_lift_speed: int
    send_delay_seconds: float
    suction_delay_seconds: float
    stack_pick_press_offset: float
    emit_speed_command: bool
    lift_after_release: bool
    start_with_suction_off: bool
    end_with_suction_off: bool


def format_number(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def move_command(x: float, y: float, z: float, duration: float) -> str:
    return "DescartesPoint_{},{},{},{},".format(
        format_number(x),
        format_number(y),
        format_number(z),
        format_number(duration),
    )


def vertical_move_command(x: float, y: float, z: float, cfg: StackConfig) -> str:
    return move_command(x, y, z, cfg.vertical_move_time)


def load_config(path: Path) -> StackConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    target = raw.get("target") or {}
    return StackConfig(
        pickup_z=float(raw.get("pickup_z", 80.9)),
        dice_height=float(raw.get("dice_height", 16.0)),
        target_x=float(target.get("x", 205.0)),
        target_y=float(target.get("y", 22.0)),
        target_z=float(target.get("z", raw.get("pickup_z", 80.9))),
        safe_margin_z=float(raw.get("safe_margin_z", 30.0)),
        move_time=float(raw.get("move_time", 100)),
        vertical_move_time=float(raw.get("vertical_move_time", raw.get("move_time", 100))),
        release_lift_time=float(raw.get("release_lift_time", raw.get("move_time", 100) * 2)),
        speed=int(float(raw.get("speed", 100))),
        release_lift_speed=int(float(raw.get("release_lift_speed", 10))),
        send_delay_seconds=float(raw.get("send_delay_seconds", 0.4)),
        suction_delay_seconds=float(raw.get("suction_delay_seconds", 0.8)),
        stack_pick_press_offset=float(raw.get("stack_pick_press_offset", -2.0)),
        emit_speed_command=bool(raw.get("emit_speed_command", True)),
        lift_after_release=bool(raw.get("lift_after_release", True)),
        start_with_suction_off=bool(raw.get("start_with_suction_off", True)),
        end_with_suction_off=bool(raw.get("end_with_suction_off", True)),
    )


def parse_inline_coords(text: str) -> list[Point2D]:
    points: list[Point2D] = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [item.strip() for item in chunk.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Bad coordinate item: {chunk!r}")
        points.append(Point2D(float(parts[0]), float(parts[1])))
    return points


def parse_coords_file(path: Path) -> list[Point2D]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        points: list[Point2D] = []
        for item in data:
            if isinstance(item, dict):
                if "world_x" in item and "world_y" in item:
                    points.append(Point2D(float(item["world_x"]), float(item["world_y"])))
                else:
                    points.append(Point2D(float(item["x"]), float(item["y"])))
            else:
                points.append(Point2D(float(item[0]), float(item[1])))
        return points

    first_line = text.splitlines()[0].strip().lower()
    has_named_header = "x" in [item.strip() for item in first_line.split(",")] and "y" in [
        item.strip() for item in first_line.split(",")
    ]
    if has_named_header:
        rows = list(csv.DictReader(text.splitlines()))
        points = []
        for row in rows:
            lower = {key.lower().strip(): value for key, value in row.items() if key is not None}
            points.append(Point2D(float(lower["x"]), float(lower["y"])))
        return points

    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        points.append(Point2D(float(parts[0]), float(parts[1])))
    return points


def generate_commands(points: Iterable[Point2D], cfg: StackConfig) -> list[str]:
    commands: list[str] = []
    if cfg.emit_speed_command:
        commands.append(f"Speed_{cfg.speed},")
    if cfg.start_with_suction_off:
        commands.append("Suction_0,")

    for index, point in enumerate(points):
        place_z = cfg.target_z + index * cfg.dice_height
        safe_z = max(cfg.pickup_z, place_z) + cfg.safe_margin_z

        pick_start = []
        if index > 0 or not cfg.start_with_suction_off:
            pick_start.append("Suction_0,")

        commands.extend(
            pick_start
            + [
                move_command(point.x, point.y, safe_z, cfg.move_time),
                vertical_move_command(point.x, point.y, cfg.pickup_z, cfg),
                "Suction_1,",
                vertical_move_command(point.x, point.y, safe_z, cfg),
                move_command(cfg.target_x, cfg.target_y, safe_z, cfg.move_time),
                vertical_move_command(cfg.target_x, cfg.target_y, place_z, cfg),
                "Suction_0,",
            ]
        )
        if cfg.lift_after_release:
            next_safe_z = max(cfg.pickup_z, place_z + cfg.dice_height) + cfg.safe_margin_z
            commands.append(f"Speed_{cfg.release_lift_speed},")
            commands.append(move_command(cfg.target_x, cfg.target_y, next_safe_z, cfg.release_lift_time))
            commands.append(f"Speed_{cfg.speed},")

    if cfg.end_with_suction_off:
        commands.append("Suction_0,")

    return commands


def send_commands(commands: Iterable[str], port: str, baudrate: int, cfg: StackConfig) -> None:
    try:
        import serial
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit("Missing dependency: pyserial. Install with `pip install pyserial`.") from exc

    with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as ser:
        for command in commands:
            print(command)
            ser.write(f"{command}\r\n".encode("utf-8"))
            delay = cfg.suction_delay_seconds if command.startswith("Suction_") else cfg.send_delay_seconds
            time.sleep(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or send serial commands to pick six dice and stack them at a fixed target."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"YAML config path. Default: {CONFIG_PATH}",
    )
    parser.add_argument(
        "--coords",
        help='Inline 2D world coordinates, e.g. "188.1,-17;223.1,-18.6;222.6,0.9".',
    )
    parser.add_argument(
        "--coords-file",
        type=Path,
        help="CSV/JSON/TXT coordinate file. CSV may contain x,y columns or value,x,y,color columns.",
    )
    parser.add_argument("--send", action="store_true", help="Send commands to the serial port after printing them.")
    parser.add_argument("--port", help="Serial port used with --send, for example COM3.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate used with --send.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.coords and args.coords_file:
        parser.error("Use only one of --coords or --coords-file.")
    if args.coords:
        points = parse_inline_coords(args.coords)
    elif args.coords_file:
        points = parse_coords_file(args.coords_file)
    else:
        points = [Point2D(float(row["x"]), float(row["y"])) for row in csv.DictReader(sys.stdin)]

    if len(points) != 6:
        raise SystemExit(f"Expected 6 dice coordinates, got {len(points)}.")

    commands = generate_commands(points, cfg)
    if args.send:
        if not args.port:
            parser.error("--send requires --port, for example `--send --port COM3`.")
        send_commands(commands, args.port, args.baudrate, cfg)
    else:
        print("\n".join(commands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
