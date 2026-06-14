"""AMR movement visualizer — loads from simulation_log/ and replays with interactive controls.

Usage:
    python visualizer.py                          # latest log in simulation_log/
    python visualizer.py --log simulation_log/amr_motion_20260505_123456.json
    python visualizer.py --interval-ms 120
    python visualizer.py --speed 10 --interval-ms 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

# Force an interactive backend so FuncAnimation timers actually fire.
# Try TkAgg (default on Windows) first, then Qt as fallback.
for _backend in ("TkAgg", "Qt5Agg", "QtAgg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except (ImportError, ValueError):
        continue

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AMR_LIST, MAP_PATH

LOG_DIR = ROOT_DIR / "simulation_log"
MIN_FRAME_SPEED = 1.0
MAX_FRAME_SPEED = 100.0
MIN_PLAYBACK_INTERVAL_MS = 15
BOTTLENECK_LABEL_ZORDER = 11
BOTTLENECK_LABEL_HOVER_ZORDER = 40
DEFAULT_TRACE_MAX_POINTS = 600

# Temporary AMR footprint for visualization only.
# Map cell size is 0.2 m, so 4 x 3 cells renders as roughly 0.8 m x 0.6 m.
AMR_BODY_LENGTH_CELLS = 4.0
AMR_BODY_WIDTH_CELLS = 3.0
CARRYING_BADGE_COLORS = {
    "실대차": "#1565C0",
    "공대차": "#EF6C00",
}

_DEFAULT_PALETTE = [
    "#E53935", "#3949AB", "#00897B", "#F57C00", "#8E24AA", "#00838F",
]
AMR_COLORS = {
    cfg["AMR_id"]: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
    for i, cfg in enumerate(AMR_LIST)
}
BLOCKED_PATH_EDGES = {
    frozenset(("N170", "N281")),
    frozenset(("N271", "N288")),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_latest_log(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob("amr_motion_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_frames(records: list[dict]) -> list[tuple[float, list[dict]]]:
    by_time: dict[float, list] = defaultdict(list)
    for r in records:
        by_time[float(r["time"])].append(r)
    return [(t, by_time[t]) for t in sorted(by_time)]


def heading_degrees(xs, ys, count: int) -> float:
    if count <= 1:
        return 0.0

    current_x = float(xs[count - 1])
    current_y = float(ys[count - 1])
    start = max(0, count - 80)

    for index in range(count - 2, start - 1, -1):
        prev_x = float(xs[index])
        prev_y = float(ys[index])
        dx = current_x - prev_x
        dy = current_y - prev_y
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx))

    return 0.0


def carrying_type_from_record(record: dict) -> str | None:
    carrying_type = record.get("carrying_type")
    if carrying_type in CARRYING_BADGE_COLORS:
        return carrying_type

    task_step = record.get("task_step")
    if task_step == "recovery_to_empty":
        return "공대차"
    if task_step == "from_to":
        if record.get("load_type") == "공대차":
            return "공대차"
        return "실대차"
    return None


def build_trace_series(
    frames: list[tuple[float, list[dict]]], amr_ids: list[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[int]]]:
    traces: dict[str, tuple[list[float], list[float]]] = {
        a: ([], []) for a in amr_ids
    }
    counts: dict[str, list[int]] = {a: [] for a in amr_ids}

    for _, items in frames:
        for item in items:
            aid = item["AMR_id"]
            if aid in traces:
                traces[aid][0].append(item["x"] + 0.5)
                traces[aid][1].append(item["y"] + 0.5)

        for aid in amr_ids:
            counts[aid].append(len(traces[aid][0]))

    trace_arrays = {
        aid: (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        for aid, (xs, ys) in traces.items()
    }
    return trace_arrays, counts


def precompute_positions(
    frames: list[tuple[float, list[dict]]], amr_ids: list[str]
) -> list[dict[str, tuple[float, float, dict]]]:
    running: dict[str, tuple[float, float, dict] | None] = {a: None for a in amr_ids}
    snapshots = []
    for _, items in frames:
        for item in items:
            aid = item["AMR_id"]
            if aid in running:
                running[aid] = (item["x"] + 0.5, item["y"] + 0.5, item)
        snapshots.append({a: pos for a, pos in running.items() if pos is not None})
    return snapshots


# ---------------------------------------------------------------------------
# Map drawing
# ---------------------------------------------------------------------------

def _iter_map_cells(map_data: dict):
    for key in ("wh_locations", "ml_nodes", "ps_nodes"):
        for record in map_data.get(key, []):
            coord = record.get("coordinates") or {}
            if coord.get("x") is not None and coord.get("y") is not None:
                yield float(coord["x"]), float(coord["y"])

    for node in map_data.get("amr_path_nodes", []):
        if node.get("x") is not None and node.get("y") is not None:
            yield float(node["x"]), float(node["y"])

    for region in map_data.get("regions", []):
        for cell in region.get("cells", []):
            if len(cell) >= 2:
                yield float(cell[0]), float(cell[1])


def _visible_bounds(map_data: dict, margin: int = 2) -> tuple[float, float, float, float]:
    meta = map_data["metadata"]
    width = int(meta["grid"]["width"])
    height = int(meta["grid"]["height"])
    points = list(_iter_map_cells(map_data))
    if not points:
        return 0, width, 0, height

    xs, ys = zip(*points)
    x_min = max(0, int(np.floor(min(xs))) - margin)
    x_max = min(width, int(np.ceil(max(xs))) + 1 + margin)
    y_min = max(0, int(np.floor(min(ys))) - margin)
    y_max = min(height, int(np.ceil(max(ys))) + 1 + margin)
    return x_min, x_max, y_min, y_max


def _path_node_xy(node: dict) -> tuple[float, float]:
    return float(node["x"]) + 0.5, float(node["y"]) + 0.5


def _location_points(map_data: dict) -> dict:
    points = {}
    for key in ("wh_locations", "ml_nodes", "ps_nodes"):
        for record in map_data.get(key, []):
            loc_id = record.get("location_id")
            coord = record.get("coordinates") or {}
            if loc_id is None or coord.get("x") is None or coord.get("y") is None:
                continue
            points[loc_id] = (float(coord["x"]) + 0.5, float(coord["y"]) + 1.0)
    return points


def _path_line_segments(path_nodes: list[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    node_points = {
        (int(node["x"]), int(node["y"])): (node.get("name"), _path_node_xy(node))
        for node in path_nodes
    }
    segments = []
    for gx, gy in node_points:
        here_name, here = node_points[(gx, gy)]
        for neighbor in ((gx + 1, gy), (gx, gy + 1)):
            if neighbor in node_points:
                neighbor_name, neighbor_xy = node_points[neighbor]
                if frozenset((here_name, neighbor_name)) in BLOCKED_PATH_EDGES:
                    continue
                segments.append((here, neighbor_xy))
    return segments


def _point_pair(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _background_spec(map_data: dict) -> dict | None:
    meta = map_data.get("metadata", {})
    spec = meta.get("background_image")
    if isinstance(spec, str):
        spec = {"path": spec}
    return spec if isinstance(spec, dict) else None


def _background_path(spec: dict, map_path: Path | None) -> Path | None:
    raw_path = spec.get("path") or spec.get("file")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    base_dir = Path(map_path).resolve().parent if map_path else ROOT_DIR / "modeling" / "map"
    return (base_dir / path).resolve()


def _background_adjustment(spec: dict) -> dict[str, float]:
    raw = spec.get("adjustment", {})
    if not isinstance(raw, dict):
        raw = {}

    def value(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "offset_x": value("offset_x", 0.0),
        "offset_y": value("offset_y", 0.0),
        "scale_x": value("scale_x", 1.0),
        "scale_y": value("scale_y", 1.0),
    }


def _adjust_grid_points(
    grid_points: list[tuple[float, float]],
    spec: dict,
) -> list[tuple[float, float]]:
    if not grid_points:
        return grid_points

    adjustment = _background_adjustment(spec)
    xs = [point[0] for point in grid_points]
    ys = [point[1] for point in grid_points]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    offset_x = adjustment["offset_x"]
    offset_y = adjustment["offset_y"]
    scale_x = adjustment["scale_x"]
    scale_y = adjustment["scale_y"]

    return [
        (
            center_x + (point[0] - center_x) * scale_x + offset_x,
            center_y + (point[1] - center_y) * scale_y + offset_y,
        )
        for point in grid_points
    ]


def _background_points(spec: dict) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    calibration = spec.get("calibration", {})
    points = calibration.get("points", spec.get("points"))
    if not isinstance(points, list) or len(points) < 4:
        return None

    image_points = []
    grid_points = []
    for pair in points[:4]:
        if not isinstance(pair, dict):
            return None
        image_point = _point_pair(
            pair.get("image") or pair.get("pixel") or pair.get("source")
        )
        grid_point = _point_pair(
            pair.get("grid") or pair.get("map") or pair.get("target")
        )
        if image_point is None or grid_point is None:
            return None
        image_points.append(image_point)
        grid_points.append(grid_point)
    return image_points, grid_points


def _add_background_image(ax, map_data: dict, map_path: Path | None) -> None:
    spec = _background_spec(map_data)
    if spec is None:
        return

    image_path = _background_path(spec, map_path)
    points = _background_points(spec)
    if image_path is None or points is None or not image_path.exists():
        return

    image_points, grid_points = points
    grid_points = _adjust_grid_points(grid_points, spec)
    try:
        image = mpimg.imread(image_path)
    except OSError:
        return

    h, w = image.shape[:2]
    xs = [p[0] for p in image_points]
    ys = [p[1] for p in image_points]
    x0 = max(0, int(np.floor(min(xs))))
    x1 = min(w, int(np.ceil(max(xs))))
    y0 = max(0, int(np.floor(min(ys))))
    y1 = min(h, int(np.ceil(max(ys))))
    if x1 <= x0 or y1 <= y0:
        return

    cropped = image[y0:y1, x0:x1]
    gx = [p[0] for p in grid_points]
    gy = [p[1] for p in grid_points]
    try:
        opacity = float(spec.get("opacity", 0.38))
    except (TypeError, ValueError):
        opacity = 0.38
    opacity = max(0.0, min(1.0, opacity))

    ax.imshow(
        cropped,
        extent=(min(gx), max(gx), min(gy), max(gy)),
        origin="upper",
        alpha=opacity,
        zorder=-10,
    )


def draw_static_map(ax, map_data: dict, map_path: Path | None = None) -> dict:
    """Draw map and return {location_id: (artist, edge_color)} for dynamic depots."""
    meta = map_data["metadata"]
    width = meta["grid"]["width"]
    height = meta["grid"]["height"]

    ax.set_facecolor("#FAFAFA")
    _add_background_image(ax, map_data, map_path)
    grid_segments = (
        [((x, 0), (x, height)) for x in range(width + 1)]
        + [((0, y), (width, y)) for y in range(height + 1)]
    )
    ax.add_collection(
        LineCollection(
            grid_segments,
            colors="#E5E5E5",
            linewidths=0.4,
            zorder=0,
        )
    )

    region_rects = [
        patches.Rectangle((cell[0], cell[1]), 1, 1)
        for region in map_data.get("regions", [])
        for cell in region.get("cells", [])
        if len(cell) >= 2
    ]
    if region_rects:
        ax.add_collection(
            PatchCollection(
                region_rects,
                facecolor="#C8E6C9",
                edgecolor="none",
                alpha=0.18,
                zorder=1,
                match_original=False,
            )
        )

    path_nodes = [
        node
        for node in map_data.get("amr_path_nodes", [])
        if node.get("x") is not None and node.get("y") is not None
    ]
    if path_nodes:
        path_segments = _path_line_segments(path_nodes)
        if path_segments:
            ax.add_collection(
                LineCollection(
                    path_segments,
                    colors="#D32F2F",
                    linewidths=2.0,
                    alpha=0.85,
                    capstyle="round",
                    zorder=2,
                )
            )

    # ---- depots: group into a few scatter collections (one PathCollection
    # per visual style) instead of one artist per location. This keeps the
    # per-frame redraw cheap and lets _update_depots recolor in bulk. ----
    GROUP_SPECS = {
        "wh_O":     dict(marker="s", s=70,  color="#1E88E5", lw=1.0),
        "wh_other": dict(marker="s", s=70,  color="#90CAF9", lw=1.0),
        "ml_R":     dict(marker="^", s=85,  color="#E53935", lw=1.0),
        "ml_other": dict(marker="D", s=85,  color="#FB8C00", lw=1.0),
        "ps":       dict(marker="P", s=100, color="#43A047", lw=1.1),
    }
    accum = {k: {"x": [], "y": [], "ids": [], "init": []} for k in GROUP_SPECS}

    def _add_depot(group_key, record):
        coord = record.get("coordinates") or {}
        if coord.get("x") is None or coord.get("y") is None:
            return
        g = accum[group_key]
        g["x"].append(coord["x"] + 0.5)
        g["y"].append(coord["y"] + 0.5)
        g["ids"].append(record.get("location_id"))
        g["init"].append(1 if record.get("occupied") else 0)

    for loc in map_data.get("wh_locations", []):
        _add_depot("wh_O" if loc.get("rack_type") == "O" else "wh_other", loc)
    for node in map_data.get("ml_nodes", []):
        _add_depot("ml_R" if node.get("rack_type") == "R" else "ml_other", node)
    for node in map_data.get("ps_nodes", []):
        _add_depot("ps", node)

    depot_index: dict = {}
    groups: dict = {}
    empty_rgba = np.array(mcolors.to_rgba("#FFFFFF", 1.0), dtype=float)

    for key, spec in GROUP_SPECS.items():
        g = accum[key]
        n = len(g["x"])
        if n == 0:
            continue
        occupied_rgba = np.array(mcolors.to_rgba(spec["color"], 1.0), dtype=float)
        facecolors = np.empty((n, 4), dtype=float)
        for i, occ in enumerate(g["init"]):
            facecolors[i] = occupied_rgba if occ else empty_rgba
        sc = ax.scatter(
            g["x"], g["y"],
            s=spec["s"], marker=spec["marker"],
            facecolors=facecolors, edgecolors=spec["color"],
            linewidths=spec["lw"], zorder=3,
        )
        groups[key] = {
            "collection": sc,
            "facecolors": facecolors,
            "occupied_rgba": occupied_rgba,
            "empty_rgba": empty_rgba,
        }
        for i, loc_id in enumerate(g["ids"]):
            if loc_id is not None:
                depot_index[loc_id] = (key, i)

    depot_artists = {"depot_index": depot_index, "groups": groups}

    x_min, x_max, y_min, y_max = _visible_bounds(map_data)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("AMR Movement Replay", fontsize=11)

    return depot_artists


def _initial_occupancy(map_data: dict) -> dict:
    occ = {}
    for key in ("wh_locations", "ml_nodes", "ps_nodes"):
        for record in map_data.get(key, []):
            loc_id = record.get("location_id")
            if loc_id is None:
                continue
            occ[loc_id] = 1 if record.get("occupied") else 0
    return occ


def build_depot_timeline(
    initial: dict, events: list[dict]
) -> dict:
    """Build per-location sorted (time, occupied) list including initial state at t=-inf."""
    timeline: dict[str, list[tuple[float, int]]] = {
        loc_id: [(float("-inf"), occ)] for loc_id, occ in initial.items()
    }
    for ev in sorted(events, key=lambda e: float(e.get("time") or 0)):
        loc_id = ev.get("location_id")
        if loc_id is None:
            continue
        timeline.setdefault(loc_id, [(float("-inf"), 0)]).append(
            (float(ev.get("time") or 0), 1 if ev.get("occupied") else 0)
        )
    return timeline


def occupancy_at(timeline_entry: list[tuple[float, int]], t: float) -> int:
    occ = 0
    for tt, val in timeline_entry:
        if tt <= t:
            occ = val
        else:
            break
    return occ


def build_bottleneck_timeline(events: list[dict]) -> dict:
    def _int_value(value):
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    timeline: dict[str, list[tuple[float, int, int]]] = {}
    for ev in sorted(events, key=lambda e: float(e.get("time") or 0)):
        loc_id = ev.get("location_id")
        if loc_id is None:
            continue
        timeline.setdefault(loc_id, [(float("-inf"), 0, 0)]).append(
            (
                float(ev.get("time") or 0),
                _int_value(ev.get("current")),
                _int_value(ev.get("total")),
            )
        )
    return timeline


def bottleneck_at(
    timeline_entry: list[tuple[float, int, int]], t: float
) -> tuple[int, int]:
    current, total = 0, 0
    for tt, cur, tot in timeline_entry:
        if tt <= t:
            current, total = cur, tot
        else:
            break
    return current, total


def style_amr_marker(artist, color: str, amr_status: str | None):
    if amr_status == "UNDISPATCHED":
        artist.set_facecolor(mcolors.to_rgba(color, 0.0))
        artist.set_edgecolor(mcolors.to_rgba(color, 1.0))
        artist.set_linewidth(2.2)
    else:
        artist.set_facecolor(mcolors.to_rgba(color, 0.92))
        artist.set_edgecolor(mcolors.to_rgba("#FFFFFF", 1.0))
        artist.set_linewidth(1.6)


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------

class Replayer:
    def __init__(
        self,
        map_data: dict,
        records: list[dict],
        log_path: Path,
        depot_events: list[dict] | None = None,
        bottleneck_events: list[dict] | None = None,
        interval_ms: int = 180,
        initial_speed: float = 1.0,
        trace_max_points: int = DEFAULT_TRACE_MAX_POINTS,
        map_path: Path | None = None,
    ):
        self.map_data = map_data
        self.records = records
        self.log_path = log_path
        self.map_path = Path(map_path) if map_path is not None else None
        self.interval_ms = max(1, int(interval_ms))
        self.frame_speed = self._clamp_frame_speed(initial_speed)
        self.trace_max_points = max(0, int(trace_max_points))
        self.depot_events = depot_events or []
        self.bottleneck_events = bottleneck_events or []

        self.frames = build_frames(records)
        self.amr_ids = sorted({r["AMR_id"] for r in records})
        self.frame_times = [t for t, _ in self.frames]
        self.trace_series, self.trace_counts = build_trace_series(
            self.frames, self.amr_ids
        )
        self.position_snapshots = precompute_positions(self.frames, self.amr_ids)
        self.initial_occupancy = _initial_occupancy(map_data)
        self.depot_timeline = build_depot_timeline(
            self.initial_occupancy, self.depot_events
        )
        self.bottleneck_timeline = build_bottleneck_timeline(self.bottleneck_events)

        self.current_frame = 0
        self._frame_advance_remainder = 0.0
        self._last_drawn_frame: int | None = None
        self.playing = True
        self._slider_busy = False
        self.path_visible = {a: True for a in self.amr_ids}
        self.depot_artists: dict = {}
        self._depot_prev_occ: dict = {}
        self.bottleneck_labels: dict = {}
        self.hovered_bottleneck_loc: str | None = None

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------

    def build(self):
        self.fig = plt.figure(figsize=(20, 8))
        self.fig.patch.set_facecolor("#F5F5F5")

        # Map (left, large)
        self.ax_map = self.fig.add_axes([0.04, 0.30, 0.64, 0.58])
        # Log panel (right)
        self.ax_log = self.fig.add_axes([0.82, 0.30, 0.16, 0.58])
        # CheckButtons (middle column)
        n = len(self.amr_ids)
        ch_h = max(0.06 * n + 0.06, 0.20)
        ch_b = 0.30 + (0.58 - ch_h) / 2
        self.ax_check = self.fig.add_axes([0.70, ch_b, 0.08, ch_h])
        # Slider
        self.ax_slider = self.fig.add_axes([0.11, 0.18, 0.58, 0.035])
        # Buttons
        self.ax_btn_play = self.fig.add_axes([0.11, 0.06, 0.12, 0.065])
        self.ax_btn_start = self.fig.add_axes([0.26, 0.06, 0.12, 0.065])
        # Goto-time TextBox
        self.ax_txt_goto = self.fig.add_axes([0.45, 0.06, 0.10, 0.065])
        # Frame speed control
        self.ax_speed = self.fig.add_axes([0.68, 0.075, 0.24, 0.035])

        # ---- map ----
        self.depot_artists = draw_static_map(self.ax_map, self.map_data, self.map_path)
        self._build_bottleneck_labels()
        self.time_text = self.ax_map.text(
            0.02, 0.98, "",
            transform=self.ax_map.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.9),
            zorder=10,
        )

        # ---- AMR artists ----
        self.amr_artists: dict = {}
        self.amr_labels: dict = {}
        self.amr_carry_labels: dict = {}
        self.trace_lines: dict = {}

        for aid in self.amr_ids:
            color = AMR_COLORS.get(aid, "#6D4C41")
            sc = patches.Rectangle(
                (-AMR_BODY_LENGTH_CELLS / 2, -AMR_BODY_WIDTH_CELLS / 2),
                AMR_BODY_LENGTH_CELLS,
                AMR_BODY_WIDTH_CELLS,
                facecolor=color,
                edgecolor="white",
                linewidth=1.6,
                zorder=8,
                visible=False,
                clip_on=True,
            )
            self.ax_map.add_patch(sc)
            lbl = self.ax_map.text(
                0, 0, aid, ha="center", va="bottom",
                fontsize=8, color=color, weight="bold", zorder=9, visible=False,
                clip_on=True,
            )
            carry_lbl = self.ax_map.text(
                0, 0, "", ha="center", va="center",
                fontsize=7, weight="bold", zorder=10, visible=False,
                bbox=dict(facecolor="#FFFFFF", edgecolor="#263238", alpha=0.88, pad=0.8),
                clip_on=True,
            )
            line, = self.ax_map.plot(
                [], [], color=color, linewidth=1.5, alpha=0.45, zorder=7,
            )
            self.amr_artists[aid] = sc
            self.amr_labels[aid] = lbl
            self.amr_carry_labels[aid] = carry_lbl
            self.trace_lines[aid] = line

        # ---- CheckButtons ----
        self.ax_check.set_title("Path ON/OFF", fontsize=8, pad=4)
        self._check = CheckButtons(
            self.ax_check, self.amr_ids, [True] * n,
        )
        for lbl_txt, aid in zip(self._check.labels, self.amr_ids):
            lbl_txt.set_color(AMR_COLORS.get(aid, "#333333"))
            lbl_txt.set_fontsize(8)
        self._check.on_clicked(self._on_check_clicked)

        # ---- Log panel ----
        self.ax_log.set_facecolor("#FAFAFA")
        self.ax_log.axis("off")
        self.ax_log.set_title(
            f"Log: {self.log_path.name}", fontsize=8, color="#555555", pad=4,
        )
        self._log_text = self.ax_log.text(
            0.03, 0.98, "",
            transform=self.ax_log.transAxes,
            va="top", ha="left",
            fontsize=7, family="monospace", color="#222222",
            clip_on=True,
        )

        # ---- Slider ----
        max_fi = max(0, len(self.frames) - 1)
        self._slider = Slider(
            self.ax_slider, "Frame", 0, max_fi,
            valinit=0, valstep=1, color="#90CAF9",
        )
        self._slider.on_changed(self._on_slider_changed)

        # ---- Buttons (plain ASCII labels) ----
        self._btn_play = Button(
            self.ax_btn_play, "Pause", color="#E3F2FD", hovercolor="#BBDEFB",
        )
        self._btn_play.on_clicked(self._on_play_pause)

        self._btn_start = Button(
            self.ax_btn_start, "Go to Start", color="#E8F5E9", hovercolor="#C8E6C9",
        )
        self._btn_start.on_clicked(self._on_go_to_start)

        # ---- Goto-time TextBox (jump to a specific simulation time t) ----
        self.fig.text(
            0.415, 0.092, "Goto t",
            ha="right", va="center", fontsize=10, color="#222222",
        )
        self._txt_goto = TextBox(
            self.ax_txt_goto, "", initial="", color="#FFFFFF",
            hovercolor="#F0F4FF",
        )
        self._txt_goto.on_submit(self._on_goto_submit)

        # ---- Frame speed Slider ----
        self._speed_slider = Slider(
            self.ax_speed,
            "Frame speed",
            MIN_FRAME_SPEED,
            MAX_FRAME_SPEED,
            valinit=self.frame_speed,
            valstep=1.0,
            color="#FFE082",
        )
        self._speed_slider.valtext.set_text(self._format_speed(self.frame_speed))
        self._speed_slider.on_changed(self._on_speed_changed)

        # ---- Keyboard navigation (left/right arrows step one frame) ----
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # ---- Title ----
        self.fig.text(
            0.34, 0.96,
            f"AMR Replay  |  {len(self.records)} records  /  {len(self.frames)} frames",
            ha="center", va="top", fontsize=10, color="#333333",
        )

        # Draw initial frame so the user sees something before animation kicks in.
        self._draw_frame(0)

        # ---- FuncAnimation playback ----
        # Pause is implemented via the self.playing flag (the timer keeps running).
        self._anim = animation.FuncAnimation(
            self.fig,
            self._animation_step,
            frames=len(self.frames),
            interval=self._playback_interval_ms(),
            blit=False,
            repeat=True,
            cache_frame_data=False,
        )

        # Force initial canvas draw so the animation event source attaches.
        self.fig.canvas.draw_idle()

        plt.show()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _build_bottleneck_labels(self):
        points = _location_points(self.map_data)
        for loc_id in self.bottleneck_timeline:
            point = points.get(loc_id)
            if point is None:
                continue
            label = self.ax_map.text(
                point[0], point[1], "",
                ha="center", va="bottom",
                fontsize=6, color="#B71C1C", weight="bold",
                bbox=dict(facecolor="#FFFFFF", edgecolor="#B71C1C", alpha=0.86, pad=0.8),
                zorder=BOTTLENECK_LABEL_ZORDER,
                visible=False,
                clip_on=True,
            )
            self.bottleneck_labels[loc_id] = label

    def _on_play_pause(self, _event):
        self.playing = not self.playing
        self._btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _on_go_to_start(self, _event):
        self.current_frame = 0
        self._frame_advance_remainder = 0.0
        self._draw_frame(0)
        self._sync_slider(0)
        self.fig.canvas.draw_idle()

    def _on_slider_changed(self, val):
        if self._slider_busy:
            return
        fi = int(val)
        self.current_frame = fi
        self._frame_advance_remainder = 0.0
        self._draw_frame(fi)
        self.fig.canvas.draw_idle()

    def _on_check_clicked(self, label):
        self.path_visible[label] = not self.path_visible[label]
        self.trace_lines[label].set_visible(self.path_visible[label])
        self.fig.canvas.draw_idle()

    def _frame_index_for_time(self, t: float) -> int:
        if not self.frames:
            return 0
        best_i = 0
        best_d = abs(self.frame_times[0] - t)
        for i in range(1, len(self.frame_times)):
            d = abs(self.frame_times[i] - t)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _seek_to_frame(self, fi: int):
        fi = max(0, min(fi, len(self.frames) - 1))
        self.playing = False
        self._btn_play.label.set_text("Play")
        self.current_frame = fi
        self._frame_advance_remainder = 0.0
        self._draw_frame(fi)
        self._sync_slider(fi)
        self.fig.canvas.draw_idle()

    def _on_goto_submit(self, text):
        text = (text or "").strip()
        if not text:
            return
        try:
            t = float(text)
        except ValueError:
            return
        self._seek_to_frame(self._frame_index_for_time(t))

    def _on_speed_changed(self, val):
        self.frame_speed = self._clamp_frame_speed(val)
        self._frame_advance_remainder = 0.0
        self._speed_slider.valtext.set_text(self._format_speed(self.frame_speed))
        if hasattr(self, "_anim") and self._anim.event_source is not None:
            self._anim.event_source.interval = self._playback_interval_ms()
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key == "right":
            self._seek_to_frame(self.current_frame + 1)
        elif event.key == "left":
            self._seek_to_frame(self.current_frame - 1)

    def _on_mouse_move(self, event):
        loc_id = self._bottleneck_label_at(event)
        if loc_id == self.hovered_bottleneck_loc:
            return

        if self.hovered_bottleneck_loc is not None:
            old_label = self.bottleneck_labels.get(self.hovered_bottleneck_loc)
            if old_label is not None:
                old_label.set_zorder(BOTTLENECK_LABEL_ZORDER)

        self.hovered_bottleneck_loc = loc_id

        if loc_id is not None:
            self.bottleneck_labels[loc_id].set_zorder(
                BOTTLENECK_LABEL_HOVER_ZORDER
            )

        self.fig.canvas.draw_idle()

    def _bottleneck_label_at(self, event) -> str | None:
        if event.inaxes is not self.ax_map or event.x is None or event.y is None:
            return None

        renderer = self.fig.canvas.get_renderer()
        candidates = []
        for loc_id, label in self.bottleneck_labels.items():
            if not label.get_visible():
                continue
            bbox = label.get_window_extent(renderer=renderer).expanded(1.08, 1.20)
            if not bbox.contains(event.x, event.y):
                continue
            lx, ly = label.get_transform().transform(label.get_position())
            distance = (lx - event.x) ** 2 + (ly - event.y) ** 2
            candidates.append((distance, loc_id))

        if not candidates:
            return None
        return min(candidates)[1]

    # ------------------------------------------------------------------
    # Animation step  (called by FuncAnimation at the base interval_ms)
    # ------------------------------------------------------------------

    def _animation_step(self, anim_frame_idx):
        # FuncAnimation drives the timer; respect pause via self.playing.
        if not self.playing:
            # Paused: the displayed frame is already drawn (slider/goto/keys
            # redraw directly). Skip the per-tick full-canvas redraw unless the
            # frame somehow changed without being drawn.
            if self._last_drawn_frame == self.current_frame:
                return []
            self._draw_frame(self.current_frame)
            self.fig.canvas.draw_idle()
            return []

        fi = self.current_frame
        self._draw_frame(fi)
        self._sync_slider(fi)
        step = self._frame_step()
        self.current_frame = (fi + step) % len(self.frames)
        return []

    def _frame_step(self) -> int:
        playback_interval = self._playback_interval_ms()
        advance = (
            self.frame_speed
            * playback_interval
            / max(1, self.interval_ms)
            + self._frame_advance_remainder
        )
        if advance < 1.0:
            self._frame_advance_remainder = 0.0
            return 1

        whole_frames = int(advance)
        self._frame_advance_remainder = advance - whole_frames
        return whole_frames

    def _sync_slider(self, fi: int):
        self._slider_busy = True
        self._slider.set_val(fi)
        self._slider_busy = False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_frame(self, fi: int):
        if not self.frames:
            return
        fi = max(0, min(fi, len(self.frames) - 1))
        time_val, items = self.frames[fi]

        current_positions = self.position_snapshots[fi]
        for aid in self.amr_ids:
            sc = self.amr_artists[aid]
            lbl = self.amr_labels[aid]
            carry_lbl = self.amr_carry_labels[aid]
            line = self.trace_lines[aid]
            color = AMR_COLORS.get(aid, "#6D4C41")

            if aid not in current_positions:
                sc.set_visible(False)
                lbl.set_visible(False)
                carry_lbl.set_visible(False)
                line.set_data([], [])
                continue

            x, y, item = current_positions[aid]
            xs, ys = self.trace_series[aid]
            count = self.trace_counts[aid][fi]
            sc.set_xy((x - AMR_BODY_LENGTH_CELLS / 2, y - AMR_BODY_WIDTH_CELLS / 2))
            sc.set_transform(
                transforms.Affine2D().rotate_deg_around(
                    x,
                    y,
                    heading_degrees(xs, ys, count),
                )
                + self.ax_map.transData
            )
            sc.set_visible(True)
            style_amr_marker(sc, color, item.get("amr_status"))
            carrying_type = carrying_type_from_record(item)
            if carrying_type:
                carry_color = CARRYING_BADGE_COLORS[carrying_type]
                sc.set_edgecolor(carry_color)
                carry_lbl.set_text(carrying_type)
                carry_lbl.set_color(carry_color)
                carry_lbl.set_position((x, y - AMR_BODY_WIDTH_CELLS / 2 - 0.55))
                carry_lbl.set_visible(True)
            else:
                carry_lbl.set_visible(False)
            lbl.set_position((x, y + AMR_BODY_WIDTH_CELLS / 2 + 0.45))
            lbl.set_visible(True)

            if self.path_visible.get(aid, True):
                start = (
                    max(0, count - self.trace_max_points)
                    if self.trace_max_points
                    else 0
                )
                line.set_data(xs[start:count], ys[start:count])
            else:
                line.set_data([], [])

        self._update_depots(time_val)
        self._update_bottlenecks(time_val)

        self.time_text.set_text(
            f"t = {time_val:.2f}   frame {fi + 1}/{len(self.frames)}   frame speed {self._format_speed(self.frame_speed)}"
        )
        self._update_log_panel(fi)
        self._last_drawn_frame = fi

    def _clamp_frame_speed(self, value: float) -> float:
        try:
            speed = float(value)
        except (TypeError, ValueError):
            speed = 1.0
        return round(max(MIN_FRAME_SPEED, min(MAX_FRAME_SPEED, speed)), 2)

    def _format_speed(self, value: float) -> str:
        return f"{float(value):.2f}x"

    def _playback_interval_ms(self) -> int:
        return max(
            MIN_PLAYBACK_INTERVAL_MS,
            int(round(self.interval_ms / max(MIN_FRAME_SPEED, self.frame_speed))),
        )

    def _update_depots(self, t: float):
        groups = self.depot_artists.get("groups")
        depot_index = self.depot_artists.get("depot_index")
        if not groups or not depot_index:
            return

        dirty_groups = set()
        prev = self._depot_prev_occ
        for loc_id, (group_key, idx) in depot_index.items():
            entry = self.depot_timeline.get(loc_id)
            if entry is None:
                continue
            occ = occupancy_at(entry, t)
            if prev.get(loc_id) == occ:
                continue
            prev[loc_id] = occ
            grp = groups[group_key]
            grp["facecolors"][idx] = (
                grp["occupied_rgba"] if occ else grp["empty_rgba"]
            )
            dirty_groups.add(group_key)

        for group_key in dirty_groups:
            grp = groups[group_key]
            grp["collection"].set_facecolors(grp["facecolors"])

    def _hide_bottleneck_label(self, loc_id: str, label):
        label.set_visible(False)
        label.set_zorder(BOTTLENECK_LABEL_ZORDER)
        if self.hovered_bottleneck_loc == loc_id:
            self.hovered_bottleneck_loc = None

    def _update_bottlenecks(self, t: float):
        for loc_id, label in self.bottleneck_labels.items():
            entry = self.bottleneck_timeline.get(loc_id)
            if entry is None:
                self._hide_bottleneck_label(loc_id, label)
                continue
            current, total = bottleneck_at(entry, t)
            if current == 0 and total == 0:
                self._hide_bottleneck_label(loc_id, label)
                continue
            label.set_text(f"({current}/{total})")
            label.set_visible(True)

    def _update_log_panel(self, fi: int):
        start = max(0, fi - 24)
        lines = []
        for i in range(start, fi + 1):
            t, items = self.frames[i]
            for entry in items:
                mark = "V" if entry.get("result_status") == "SUCCESS" else "-"
                try:
                    speed = f"{float(entry.get('speed')):.2f}m/s"
                except (TypeError, ValueError):
                    speed = "--m/s"
                try:
                    x_text = f"{float(entry['x']):.1f}"
                    y_text = f"{float(entry['y']):.1f}"
                except (TypeError, ValueError):
                    x_text = str(entry.get("x"))
                    y_text = str(entry.get("y"))
            lines.append(
                f"[{mark}] t={t:<6.1f} {entry['AMR_id']}\n"
                f"     ({x_text:>5},{y_text:>5})"
                f" {entry.get('task_step','')}\n"
                f"     v={speed} carrying={carrying_type_from_record(entry) or '미적재'}\n"
                f"     {entry.get('result_status','')}"
            )
        self._log_text.set_text("\n".join(lines[-8:]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AMR movement logs.")
    parser.add_argument(
        "--log", default=None,
        help="Path to amr_motion_*.json. Defaults to latest in simulation_log/.",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Path to map JSON. Defaults to log metadata or the selected config map.",
    )
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument(
        "--speed",
        "--frame-speed",
        dest="speed",
        type=float,
        default=1.0,
        help="Initial frame playback speed multiplier, clamped to 1.00-100.00.",
    )
    parser.add_argument(
        "--trace-window",
        type=int,
        default=DEFAULT_TRACE_MAX_POINTS,
        help="Number of recent points to draw per AMR trace. Use 0 for full traces.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_path = Path(args.log) if args.log else find_latest_log(LOG_DIR)
    if log_path is None:
        print(f"No log files found in {LOG_DIR}. Run run_and_visualize.py first.")
        sys.exit(1)

    raw = load_json(log_path)
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}

    if args.map:
        map_path = Path(args.map)
    elif metadata.get("map_file"):
        map_path = ROOT_DIR / "modeling" / "map" / metadata["map_file"]
    else:
        map_path = MAP_PATH

    print(f"Map : {map_path}")
    print(f"Log : {log_path}")

    map_data = load_json(map_path)

    if isinstance(raw, dict):
        records = raw.get("amr_motion", [])
        depot_events = raw.get("depot_events", [])
        bottleneck_events = raw.get("bottleneck_events", [])
    else:
        records = raw
        depot_events = []
        bottleneck_events = []

    if not records:
        print("Log file contains no records.")
        sys.exit(1)

    print(
        f"Loaded {len(records)} AMR records / {len(build_frames(records))} frames"
        f" / {len(depot_events)} depot events"
        f" / {len(bottleneck_events)} bottleneck events."
    )
    Replayer(
        map_data, records, log_path,
        depot_events=depot_events,
        bottleneck_events=bottleneck_events,
        interval_ms=args.interval_ms,
        initial_speed=args.speed,
        trace_max_points=args.trace_window,
        map_path=map_path,
    ).build()


if __name__ == "__main__":
    main()
