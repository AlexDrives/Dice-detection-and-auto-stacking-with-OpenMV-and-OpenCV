from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from dice_stack_commands import (
    CONFIG_PATH,
    Point2D,
    StackConfig,
    load_config,
    move_command,
    parse_coords_file,
    parse_inline_coords,
    send_commands,
    vertical_move_command,
)


def generate_restore_commands(points: list[Point2D], cfg: StackConfig) -> list[str]:
    commands: list[str] = []
    if cfg.emit_speed_command:
        commands.append(f"Speed_{cfg.speed},")
    if cfg.start_with_suction_off:
        commands.append("Suction_0,")

    total = len(points)
    for reverse_index, point in enumerate(reversed(points)):
        stack_index = total - 1 - reverse_index
        stack_z = cfg.target_z + stack_index * cfg.dice_height
        stack_pick_z = stack_z + cfg.stack_pick_press_offset
        safe_z = max(cfg.pickup_z, stack_z) + cfg.safe_margin_z

        if reverse_index > 0 or not cfg.start_with_suction_off:
            commands.append("Suction_0,")

        commands.extend(
            [
                move_command(cfg.target_x, cfg.target_y, safe_z, cfg.move_time),
                vertical_move_command(cfg.target_x, cfg.target_y, stack_pick_z, cfg),
                "Suction_1,",
                vertical_move_command(cfg.target_x, cfg.target_y, safe_z, cfg),
                move_command(point.x, point.y, safe_z, cfg.move_time),
                vertical_move_command(point.x, point.y, cfg.pickup_z, cfg),
                "Suction_0,",
            ]
        )

        if cfg.lift_after_release:
            commands.append(f"Speed_{cfg.release_lift_speed},")
            commands.append(move_command(point.x, point.y, safe_z, cfg.release_lift_time))
            commands.append(f"Speed_{cfg.speed},")

    if cfg.end_with_suction_off:
        commands.append("Suction_0,")

    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or send serial commands to pick dice from the stack and place them back."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"YAML config path. Default: {CONFIG_PATH}",
    )
    parser.add_argument(
        "--coords",
        help='Inline 2D world coordinates, e.g. "188.1,-17;188.1,0;190,-30".',
    )
    parser.add_argument(
        "--coords-file",
        type=Path,
        help="CSV/JSON/TXT coordinate file. JSON with world_x/world_y is supported.",
    )
    parser.add_argument("--send", action="store_true", help="Send commands to the serial port after printing them.")
    parser.add_argument("--port", help="Serial port used with --send, for example COM15.")
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

    commands = generate_restore_commands(points, cfg)
    if args.send:
        if not args.port:
            parser.error("--send requires --port, for example `--send --port COM15`.")
        send_commands(commands, args.port, args.baudrate, cfg)
    else:
        print("\n".join(commands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
