"""Replay AMR movement on the existing map without changing simulation code.

Usage:
    python modeling/map/replay_amr_map.py
    python modeling/map/replay_amr_map.py --max-time 300
    python modeling/map/replay_amr_map.py --log-json modeling/map/amr_motion_log.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SimulationEngine.SimulationEngine import SimulationEngine
from SimulationEngine.Utility.Configurator import Configurator
from modeling.outmost import Outmost
from config import AMR_LIST, MAP_PATH


DEFAULT_MAP_PATH = MAP_PATH
DEFAULT_LOG_PATH = ROOT_DIR / "modeling" / "map" / "amr_motion_log.json"

AMR_CONFIGS = [dict(item) for item in AMR_LIST]

AMR_COLORS = {
    "AMR_01": "#E53935",
    "AMR_02": "#3949AB",
    "AMR_03": "#00897B",
    "AMR_04": "#8E24AA",
}


def load_map(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def build_location_index(map_data: dict) -> dict:
    index = {}
    for key in ("wh_locations", "ml_nodes", "ps_nodes"):
        for record in map_data.get(key, []):
            location_id = record.get("location_id")
            if location_id is not None:
                index[location_id] = record
    return index


def point_from_location(value, location_index: dict):
    if value is None:
        return None

    if isinstance(value, str):
        value = location_index.get(value)

    if isinstance(value, dict):
        coordinates = value.get("coordinates")
        if isinstance(coordinates, dict):
            return coordinates.get("x"), coordinates.get("y")
        if value.get("x") is not None and value.get("y") is not None:
            return value.get("x"), value.get("y")

    return None


def make_configuration() -> Configurator:
    config = Configurator()
    config.addConfiguration("AMR_LIST", [dict(item) for item in AMR_CONFIGS])
    return config


def find_model(engine: SimulationEngine, model_id: str):
    for model in engine.models:
        if getattr(model, "getModelID", lambda: None)() == model_id:
            return model
    return None


def run_and_collect(max_time: float, map_data: dict, quiet: bool = True) -> list[dict]:
    location_index = build_location_index(map_data)
    records = []

    outmost = Outmost(make_configuration())
    engine = SimulationEngine()
    engine.setOutmostModel(outmost)

    analyzer = find_model(engine, "Analyzer")
    if analyzer is None:
        raise RuntimeError("Analyzer model was not found.")

    original_transition = analyzer.funcExternalTransition

    def record_transition(port, event):
        if port == "physical_result_from_amr":
            point = point_from_location(event.current_location, location_index)
            if point is not None:
                x, y = point
                records.append(
                    {
                        "time": event.timestamp,
                        "AMR_id": event.AMR_id,
                        "x": x,
                        "y": y,
                        "task_step": event.task_step,
                        "move_num": event.move_num,
                        "amr_status": event.amr_status,
                        "result_status": event.result_status,
                    }
                )
        original_transition(port, event)

    analyzer.funcExternalTransition = record_transition

    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                engine.run(
                    maxTime=max_time,
                    ta=-1,
                    visualizer=False,
                    logFileName=os.devnull,
                    logGeneral=False,
                    logActivateState=False,
                    logActivateMessage=False,
                    logActivateTA=False,
                    logStructure=False,
                )
    else:
        engine.run(
            maxTime=max_time,
            ta=-1,
            visualizer=False,
            logFileName=os.devnull,
            logGeneral=False,
            logActivateState=False,
            logActivateMessage=False,
            logActivateTA=False,
            logStructure=False,
        )

    records.sort(key=lambda item: (float(item["time"]), item["AMR_id"]))
    return records


def save_records(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def draw_static_map(ax, map_data: dict):
    meta = map_data["metadata"]
    width = meta["grid"]["width"]
    height = meta["grid"]["height"]

    ax.set_facecolor("#FAFAFA")
    for x in range(width + 1):
        ax.axvline(x, color="#E5E5E5", linewidth=0.4, zorder=0)
    for y in range(height + 1):
        ax.axhline(y, color="#E5E5E5", linewidth=0.4, zorder=0)

    for region in map_data.get("regions", []):
        for cx, cy in region.get("cells", []):
            ax.add_patch(
                patches.Rectangle(
                    (cx, cy),
                    1,
                    1,
                    facecolor="#C8E6C9",
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1,
                )
            )

    for loc in map_data.get("wh_locations", []):
        coord = loc["coordinates"]
        x, y = coord["x"] + 0.5, coord["y"] + 0.5
        rack_type = loc.get("rack_type")
        color = "#1E88E5" if rack_type == "O" else "#90CAF9"
        face = color if loc.get("occupied") else "#FFFFFF"
        ax.scatter(x, y, s=70, marker="s", c=face, edgecolors=color, linewidths=1.0, zorder=3)

    for node in map_data.get("ml_nodes", []):
        coord = node["coordinates"]
        x, y = coord["x"] + 0.5, coord["y"] + 0.5
        rack_type = node.get("rack_type")
        color = "#E53935" if rack_type == "R" else "#FB8C00"
        marker = "^" if rack_type == "R" else "D"
        ax.scatter(x, y, s=85, marker=marker, c=color, edgecolors="white", linewidths=0.8, zorder=3)

    for node in map_data.get("ps_nodes", []):
        coord = node["coordinates"]
        x, y = coord["x"] + 0.5, coord["y"] + 0.5
        face = "#43A047" if node.get("occupied") else "#FFFFFF"
        ax.scatter(x, y, s=100, marker="P", c=face, edgecolors="#43A047", linewidths=1.1, zorder=3)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("AMR movement replay")


def build_frames(records: list[dict]) -> list[tuple[float, list[dict]]]:
    by_time = defaultdict(list)
    for record in records:
        by_time[float(record["time"])].append(record)
    return [(time, by_time[time]) for time in sorted(by_time)]


def style_amr_marker(artist, color: str, amr_status: str | None):
    if amr_status == "UNDISPATCHED":
        artist.set_facecolors([mcolors.to_rgba(color, 0.0)])
        artist.set_edgecolors([mcolors.to_rgba(color, 1.0)])
        artist.set_linewidths([2.2])
        return

    artist.set_facecolors([mcolors.to_rgba(color, 0.92)])
    artist.set_edgecolors([mcolors.to_rgba("#FFFFFF", 1.0)])
    artist.set_linewidths([1.6])


def replay(map_data: dict, records: list[dict], interval_ms: int):
    if not records:
        raise RuntimeError("No AMR movement records were collected.")

    frames = build_frames(records)
    fig, ax = plt.subplots(figsize=(14, 9))
    draw_static_map(ax, map_data)

    artists = {}
    labels = {}
    traces = defaultdict(lambda: ([], []))
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.9),
        zorder=10,
    )

    for amr_id in sorted({record["AMR_id"] for record in records}):
        color = AMR_COLORS.get(amr_id, "#6D4C41")
        artist = ax.scatter([], [], s=180, marker="o", facecolors=color, edgecolors="white", linewidths=1.6, zorder=8)
        label = ax.text(0, 0, amr_id, ha="center", va="bottom", fontsize=8, color=color, weight="bold", zorder=9)
        trace_line, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.45, zorder=7)
        artists[amr_id] = (artist, trace_line)
        labels[amr_id] = label

    current_positions = {}

    def update(frame_index):
        time_value, items = frames[frame_index]
        for item in items:
            amr_id = item["AMR_id"]
            x, y = item["x"] + 0.5, item["y"] + 0.5
            current_positions[amr_id] = (x, y, item)
            traces[amr_id][0].append(x)
            traces[amr_id][1].append(y)

        changed = [time_text]
        for amr_id, (artist, trace_line) in artists.items():
            if amr_id not in current_positions:
                continue
            x, y, item = current_positions[amr_id]
            color = AMR_COLORS.get(amr_id, "#6D4C41")
            artist.set_offsets([[x, y]])
            style_amr_marker(artist, color, item.get("amr_status"))
            labels[amr_id].set_position((x, y + 0.55))
            labels[amr_id].set_visible(True)
            trace_line.set_data(traces[amr_id][0], traces[amr_id][1])
            changed.extend([artist, trace_line, labels[amr_id]])

        time_text.set_text(f"time={time_value:.2f} / frame={frame_index + 1}/{len(frames)}")
        return changed

    amr_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return amr_animation


def parse_args():
    parser = argparse.ArgumentParser(description="Replay AMR movement on the existing map.")
    parser.add_argument("--map", default=str(DEFAULT_MAP_PATH), help="Path to map JSON.")
    parser.add_argument("--max-time", type=float, default=300.0, help="Simulation max time used for replay data.")
    parser.add_argument("--interval-ms", type=int, default=180, help="Animation frame interval.")
    parser.add_argument("--log-json", default=str(DEFAULT_LOG_PATH), help="Where to save collected AMR movement records.")
    parser.add_argument("--no-show", action="store_true", help="Collect and save replay data without opening the viewer.")
    parser.add_argument("--show-sim-log", action="store_true", help="Print simulation logs while collecting data.")
    return parser.parse_args()


def main():
    args = parse_args()
    map_path = Path(args.map)
    log_path = Path(args.log_json)

    map_data = load_map(map_path)
    records = run_and_collect(args.max_time, map_data, quiet=not args.show_sim_log)
    save_records(records, log_path)

    print(f"Collected {len(records)} AMR movement records.")
    print(f"Saved movement log: {log_path}")
    if not args.no_show:
        replay(map_data, records, args.interval_ms)


if __name__ == "__main__":
    main()
