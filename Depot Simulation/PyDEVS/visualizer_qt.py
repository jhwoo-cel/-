"""Fast AMR movement replay UI based on PySide6 and pyqtgraph.

Usage:
    python visualizer_qt.py
    python visualizer_qt.py --log simulation_log/amr_motion_20260505_123456.json
    python visualizer_qt.py --speed 10 --interval-ms 100
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover - exercised only without deps.
    raise SystemExit(
        "Qt UI requires PySide6 and pyqtgraph. Install dependencies with:\n"
        "    pip install -r requirements.txt"
    ) from exc

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AMR_LIST, MAP_PATH

LOG_DIR = ROOT_DIR / "simulation_log"
DEFAULT_TRACE_MAX_POINTS = 600
MIN_FRAME_SPEED = 1.0
MAX_FRAME_SPEED = 100.0
MIN_PLAYBACK_INTERVAL_MS = 15
DEFAULT_EVENT_WINDOW_SECONDS = 60.0

# Temporary AMR footprint for visualization only.
# Map cell size is 0.2 m, so 4 x 3 cells renders as roughly 0.8 m x 0.6 m.
AMR_BODY_LENGTH_CELLS = 4.0
AMR_BODY_WIDTH_CELLS = 3.0
CARRYING_BADGE_COLORS = {
    "실대차": "#1E88E5",
    "공대차": "#90CAF9",
}

_DEFAULT_PALETTE = [
    "#E53935",
    "#3949AB",
    "#00897B",
    "#F57C00",
    "#8E24AA",
    "#00838F",
]
AMR_COLORS = {
    cfg["AMR_id"]: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
    for i, cfg in enumerate(AMR_LIST)
}
BLOCKED_PATH_EDGES = {
    frozenset(("N170", "N281")),
    frozenset(("N271", "N288")),
}

GROUP_SPECS = {
    "wh_O": dict(symbol="s", size=10, color="#1E88E5", name="WH O"),
    "wh_other": dict(symbol="s", size=10, color="#90CAF9", name="WH"),
    "ml_R": dict(symbol="t1", size=12, color="#E53935", name="ML R"),
    "ml_other": dict(symbol="d", size=12, color="#FB8C00", name="ML"),
    "ps": dict(symbol="p", size=13, color="#43A047", name="PS"),
}


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
    for record in records:
        try:
            t = float(record["time"])
        except (TypeError, ValueError, KeyError):
            continue
        by_time[t].append(record)
    return [(t, by_time[t]) for t in sorted(by_time)]


def heading_degrees(xs: np.ndarray, ys: np.ndarray, count: int) -> float:
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


def build_trace_series(
    frames: list[tuple[float, list[dict]]], amr_ids: list[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[int]]]:
    traces: dict[str, tuple[list[float], list[float]]] = {
        amr_id: ([], []) for amr_id in amr_ids
    }
    counts: dict[str, list[int]] = {amr_id: [] for amr_id in amr_ids}

    for _, items in frames:
        for item in items:
            amr_id = item.get("AMR_id")
            if amr_id in traces:
                traces[amr_id][0].append(float(item["x"]) + 0.5)
                traces[amr_id][1].append(float(item["y"]) + 0.5)
        for amr_id in amr_ids:
            counts[amr_id].append(len(traces[amr_id][0]))

    trace_arrays = {
        amr_id: (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        for amr_id, (xs, ys) in traces.items()
    }
    return trace_arrays, counts


def precompute_positions(
    frames: list[tuple[float, list[dict]]], amr_ids: list[str]
) -> list[dict[str, tuple[float, float, dict]]]:
    running: dict[str, tuple[float, float, dict] | None] = {
        amr_id: None for amr_id in amr_ids
    }
    snapshots = []
    for _, items in frames:
        for item in items:
            amr_id = item.get("AMR_id")
            if amr_id in running:
                running[amr_id] = (
                    float(item["x"]) + 0.5,
                    float(item["y"]) + 0.5,
                    item,
                )
        snapshots.append(
            {amr_id: pos for amr_id, pos in running.items() if pos is not None}
        )
    return snapshots


def build_amr_timelines(records: list[dict], amr_ids: list[str]) -> dict[str, dict]:
    timelines = {
        amr_id: {"times": [], "entries": []}
        for amr_id in amr_ids
    }
    for record in records:
        amr_id = record.get("AMR_id")
        if amr_id not in timelines:
            continue
        try:
            time_val = float(record["time"])
            x = float(record["x"]) + 0.5
            y = float(record["y"]) + 0.5
        except (TypeError, ValueError, KeyError):
            continue
        timelines[amr_id]["times"].append(time_val)
        timelines[amr_id]["entries"].append((x, y, record))

    for timeline in timelines.values():
        if not timeline["times"]:
            timeline["times"] = np.asarray([], dtype=float)
            continue
        order = np.argsort(np.asarray(timeline["times"], dtype=float), kind="stable")
        times = np.asarray(timeline["times"], dtype=float)[order]
        entries = [timeline["entries"][int(index)] for index in order]
        timeline["times"] = times
        timeline["entries"] = entries
    return timelines


def _timeline_heading(entries: list[tuple[float, float, dict]], index: int) -> float:
    if not entries:
        return 0.0

    current_x, current_y, _record = entries[index]
    start = max(0, index - 80)
    for prev_index in range(index - 1, start - 1, -1):
        prev_x, prev_y, _prev_record = entries[prev_index]
        dx = current_x - prev_x
        dy = current_y - prev_y
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx))
    return 0.0


def _can_interpolate_motion(prev_record: dict, next_record: dict) -> bool:
    if prev_record.get("result_status") == "STATION_IDLE":
        return False
    if next_record.get("result_status") == "STATION_IDLE":
        return False
    if "RUNNING" not in {prev_record.get("result_status"), next_record.get("result_status")}:
        return False
    return prev_record.get("task_step") == next_record.get("task_step")


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


def interpolated_amr_positions(
    time_val: float,
    timelines: dict[str, dict],
    max_gap: float,
) -> dict[str, tuple[float, float, dict, float]]:
    positions = {}
    for amr_id, timeline in timelines.items():
        times = timeline.get("times")
        entries = timeline.get("entries", [])
        if times is None or len(times) == 0:
            continue

        index = bisect.bisect_right(times, time_val) - 1
        if index < 0:
            continue
        if index >= len(entries):
            index = len(entries) - 1

        x, y, record = entries[index]
        heading = _timeline_heading(entries, index)

        if index + 1 < len(entries):
            next_time = float(times[index + 1])
            current_time = float(times[index])
            gap = next_time - current_time
            next_x, next_y, next_record = entries[index + 1]
            if (
                0.0 < gap <= max_gap
                and _can_interpolate_motion(record, next_record)
            ):
                ratio = max(0.0, min(1.0, (time_val - current_time) / gap))
                dx = next_x - x
                dy = next_y - y
                x = x + dx * ratio
                y = y + dy * ratio
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    heading = math.degrees(math.atan2(dy, dx))

        positions[amr_id] = (x, y, record, heading)
    return positions


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


def visible_bounds(map_data: dict, margin: int = 2) -> tuple[float, float, float, float]:
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


def path_line_arrays(path_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    node_points = {
        (int(node["x"]), int(node["y"])): (node.get("name"), _path_node_xy(node))
        for node in path_nodes
        if node.get("x") is not None and node.get("y") is not None
    }
    xs: list[float] = []
    ys: list[float] = []
    for gx, gy in node_points:
        here_name, here = node_points[(gx, gy)]
        for neighbor in ((gx + 1, gy), (gx, gy + 1)):
            if neighbor not in node_points:
                continue
            neighbor_name, neighbor_xy = node_points[neighbor]
            if frozenset((here_name, neighbor_name)) in BLOCKED_PATH_EDGES:
                continue
            xs.extend((here[0], neighbor_xy[0], math.nan))
            ys.extend((here[1], neighbor_xy[1], math.nan))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def iter_depots(map_data: dict):
    def add(group_key: str, record: dict):
        coord = record.get("coordinates") or {}
        loc_id = record.get("location_id")
        if loc_id is None or coord.get("x") is None or coord.get("y") is None:
            return None
        x = float(coord["x"]) + 0.5
        y = float(coord["y"]) + 0.5
        item = dict(record)
        item["_group_key"] = group_key
        item["_x"] = x
        item["_y"] = y
        return item

    for loc in map_data.get("wh_locations", []):
        item = add("wh_O" if loc.get("rack_type") == "O" else "wh_other", loc)
        if item is not None:
            yield item
    for node in map_data.get("ml_nodes", []):
        item = add("ml_R" if node.get("rack_type") == "R" else "ml_other", node)
        if item is not None:
            yield item
    for node in map_data.get("ps_nodes", []):
        item = add("ps", node)
        if item is not None:
            yield item


def initial_occupancy(map_data: dict) -> dict:
    occ = {}
    for record in iter_depots(map_data):
        occ[record["location_id"]] = 1 if record.get("occupied") else 0
    return occ


def build_depot_timeline(initial: dict, events: list[dict]) -> dict:
    timeline: dict[str, list[tuple[float, int]]] = {
        loc_id: [(float("-inf"), occ)] for loc_id, occ in initial.items()
    }
    for event in sorted(events, key=lambda e: float(e.get("time") or 0)):
        loc_id = event.get("location_id")
        if loc_id is None:
            continue
        timeline.setdefault(loc_id, [(float("-inf"), 0)]).append(
            (float(event.get("time") or 0), 1 if event.get("occupied") else 0)
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
    def int_value(value):
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    timeline: dict[str, list[tuple[float, int, int]]] = {}
    for event in sorted(events, key=lambda e: float(e.get("time") or 0)):
        loc_id = event.get("location_id")
        if loc_id is None:
            continue
        timeline.setdefault(loc_id, [(float("-inf"), 0, 0)]).append(
            (
                float(event.get("time") or 0),
                int_value(event.get("current")),
                int_value(event.get("total")),
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


def frame_index_for_time(frame_times: list[float], t: float) -> int:
    if not frame_times:
        return 0
    pos = bisect.bisect_left(frame_times, t)
    if pos <= 0:
        return 0
    if pos >= len(frame_times):
        return len(frame_times) - 1
    before = frame_times[pos - 1]
    after = frame_times[pos]
    return pos - 1 if abs(before - t) <= abs(after - t) else pos


def clamp_speed(value: float) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = 1.0
    return round(max(MIN_FRAME_SPEED, min(MAX_FRAME_SPEED, speed)), 2)


def load_replay_payload(log_path: Path, map_path: Path | None = None):
    raw = load_json(log_path)
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    resolved_map_path = map_path
    if resolved_map_path is None and metadata.get("map_file"):
        resolved_map_path = ROOT_DIR / "modeling" / "map" / metadata["map_file"]
    if resolved_map_path is None:
        resolved_map_path = MAP_PATH

    map_data = load_json(resolved_map_path)
    if isinstance(raw, dict):
        records = raw.get("amr_motion", [])
        depot_events = raw.get("depot_events", [])
        bottleneck_events = raw.get("bottleneck_events", [])
    else:
        records = raw
        depot_events = []
        bottleneck_events = []

    return map_data, records, depot_events, bottleneck_events, metadata, resolved_map_path


def build_timeline_events(depot_events: list[dict], bottleneck_events: list[dict]):
    events = []
    for event in bottleneck_events:
        try:
            t = float(event.get("time") or 0)
        except (TypeError, ValueError):
            continue
        events.append(
            {
                "kind": "bottleneck",
                "time": t,
                "location_id": event.get("location_id"),
                "AMR_id": event.get("AMR_id"),
                "raw": event,
            }
        )
    for event in depot_events:
        try:
            t = float(event.get("time") or 0)
        except (TypeError, ValueError):
            continue
        events.append(
            {
                "kind": "depot",
                "time": t,
                "location_id": event.get("location_id"),
                "AMR_id": event.get("AMR_id"),
                "raw": event,
            }
        )
    return sorted(events, key=lambda e: (e["time"], e["kind"], str(e.get("location_id"))))


def _point_from_pair(value, scale: tuple[float, float] = (1.0, 1.0)) -> QtCore.QPointF | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return QtCore.QPointF(float(value[0]) * scale[0], float(value[1]) * scale[1])
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
    grid_points: list[QtCore.QPointF],
    spec: dict,
) -> list[QtCore.QPointF]:
    if not grid_points:
        return grid_points

    adjustment = _background_adjustment(spec)
    xs = [point.x() for point in grid_points]
    ys = [point.y() for point in grid_points]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    offset_x = adjustment["offset_x"]
    offset_y = adjustment["offset_y"]
    scale_x = adjustment["scale_x"]
    scale_y = adjustment["scale_y"]

    return [
        QtCore.QPointF(
            center_x + (point.x() - center_x) * scale_x + offset_x,
            center_y + (point.y() - center_y) * scale_y + offset_y,
        )
        for point in grid_points
    ]


def _background_transform(
    spec: dict,
    image_scale: tuple[float, float] = (1.0, 1.0),
) -> QtGui.QTransform | None:
    calibration = spec.get("calibration", {})
    points = calibration.get("points", spec.get("points"))
    if not isinstance(points, list) or len(points) < 4:
        return None

    source = QtGui.QPolygonF()
    target_points = []
    for pair in points[:4]:
        if not isinstance(pair, dict):
            return None
        image_point = _point_from_pair(
            pair.get("image") or pair.get("pixel") or pair.get("source"),
            scale=image_scale,
        )
        grid_point = _point_from_pair(
            pair.get("grid") or pair.get("map") or pair.get("target")
        )
        if image_point is None or grid_point is None:
            return None
        source.append(image_point)
        target_points.append(grid_point)

    target = QtGui.QPolygonF()
    for point in _adjust_grid_points(target_points, spec):
        target.append(point)
    transform = QtGui.QTransform()
    if not QtGui.QTransform.quadToQuad(source, target, transform):
        return None
    return transform


def _read_background_pixmap(
    image_path: Path,
    spec: dict,
) -> tuple[QtGui.QPixmap | None, tuple[float, float], str]:
    reader = QtGui.QImageReader(str(image_path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if not source_size.isValid():
        return None, (1.0, 1.0), f"Background image size could not be read: {image_path}"

    source_w = max(1, source_size.width())
    source_h = max(1, source_size.height())
    try:
        needed_limit_mb = int(math.ceil(source_w * source_h * 4 / (1024 * 1024))) + 16
        current_limit_mb = QtGui.QImageReader.allocationLimit()
        if current_limit_mb and current_limit_mb < needed_limit_mb:
            QtGui.QImageReader.setAllocationLimit(min(1024, needed_limit_mb))
    except AttributeError:
        pass

    try:
        max_decode_pixels = int(spec.get("max_decode_pixels", 20_000_000))
    except (TypeError, ValueError):
        max_decode_pixels = 20_000_000
    max_decode_pixels = max(1_000_000, max_decode_pixels)

    decode_pixels = source_w * source_h
    scaled_w = source_w
    scaled_h = source_h
    if decode_pixels > max_decode_pixels:
        scale = math.sqrt(max_decode_pixels / decode_pixels)
        scaled_w = max(1, int(source_w * scale))
        scaled_h = max(1, int(source_h * scale))
        reader.setScaledSize(QtCore.QSize(scaled_w, scaled_h))

    image = reader.read()
    if image.isNull():
        return None, (1.0, 1.0), f"Background image could not be loaded: {reader.errorString()}"

    pixmap = QtGui.QPixmap.fromImage(image)
    if pixmap.isNull():
        return None, (1.0, 1.0), f"Background image could not be converted: {image_path}"

    image_scale = (pixmap.width() / source_w, pixmap.height() / source_h)
    status = f"Background: {image_path.name}"
    if pixmap.width() != source_w or pixmap.height() != source_h:
        status += f" ({source_w}x{source_h} -> {pixmap.width()}x{pixmap.height()})"
    return pixmap, image_scale, status


class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        map_data: dict,
        records: list[dict],
        log_path: Path,
        depot_events: list[dict] | None = None,
        bottleneck_events: list[dict] | None = None,
        metadata: dict | None = None,
        map_path: Path | None = None,
        interval_ms: int = 100,
        initial_speed: float = 1.0,
        trace_max_points: int = DEFAULT_TRACE_MAX_POINTS,
    ):
        super().__init__()
        self.interval_ms = max(1, int(interval_ms))
        self.frame_speed = clamp_speed(initial_speed)
        self.trace_max_points = max(0, int(trace_max_points))
        self.current_frame = 0
        self.current_time = 0.0
        self.current_display_positions: dict[str, tuple[float, float, dict, float]] = {}
        self.amr_timelines: dict[str, dict] = {}
        self.motion_log_interval = 0.5
        self.interpolation_max_gap = 1.25
        self._last_side_panel_wall_time = 0.0
        self._frame_advance_remainder = 0.0
        self.playing = True
        self.path_visible: dict[str, bool] = {}
        self.selected_target: tuple[str, str] | None = None
        self.selected_event: dict | None = None
        self._last_tooltip_key: tuple[str, str] | None = None
        self._timeline_click_consumed = False
        self.background_item: QtWidgets.QGraphicsPixmapItem | None = None
        self.background_status = "No background image configured."
        self.background_image_scale = (1.0, 1.0)
        self._bg_controls_busy = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._timer_tick)

        self._build_widgets()
        self.load_dataset(
            map_data,
            records,
            log_path,
            depot_events=depot_events,
            bottleneck_events=bottleneck_events,
            metadata=metadata,
            map_path=map_path,
        )

    def _build_widgets(self):
        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("AMR Replay")
        self.resize(1600, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(8)
        outer.addLayout(toolbar)

        self.btn_open = QtWidgets.QPushButton("Open Log")
        self.btn_open.clicked.connect(self._on_open_log)
        toolbar.addWidget(self.btn_open)

        self.btn_info = QtWidgets.QPushButton("Info")
        self.btn_info.clicked.connect(self._show_info)
        toolbar.addWidget(self.btn_info)

        self.btn_reset_view = QtWidgets.QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(self.reset_view)
        toolbar.addWidget(self.btn_reset_view)

        self.time_label = QtWidgets.QLabel("t = 0.00")
        self.time_label.setMinimumWidth(220)
        toolbar.addWidget(self.time_label)
        toolbar.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        self.map_plot = pg.PlotWidget()
        self.map_plot.setBackground("#FAFAFA")
        self.map_plot.showGrid(x=True, y=True, alpha=0.18)
        self.map_plot.setAspectLocked(True)
        self.map_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        splitter.addWidget(self.map_plot)

        self.side_panel = QtWidgets.QWidget()
        self.side_panel.setMinimumWidth(330)
        self.side_panel.setMaximumWidth(460)
        side_layout = QtWidgets.QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(8, 0, 0, 0)
        splitter.addWidget(self.side_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        filter_label = QtWidgets.QLabel("AMR visibility")
        filter_label.setStyleSheet("font-weight: 600;")
        side_layout.addWidget(filter_label)

        self.filter_scroll = QtWidgets.QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setMaximumHeight(150)
        self.filter_container = QtWidgets.QWidget()
        self.filter_layout = QtWidgets.QVBoxLayout(self.filter_container)
        self.filter_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_layout.setSpacing(2)
        self.filter_scroll.setWidget(self.filter_container)
        side_layout.addWidget(self.filter_scroll)

        self.full_trace_check = QtWidgets.QCheckBox("Full trace")
        self.full_trace_check.setChecked(False)
        self.full_trace_check.stateChanged.connect(lambda _state: self._draw_time(self.current_time))
        side_layout.addWidget(self.full_trace_check)

        self._build_background_controls(side_layout)

        detail_label = QtWidgets.QLabel("Selection")
        detail_label.setStyleSheet("font-weight: 600;")
        side_layout.addWidget(detail_label)

        self.detail_text = QtWidgets.QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumBlockCount(300)
        side_layout.addWidget(self.detail_text, stretch=1)

        window_row = QtWidgets.QHBoxLayout()
        window_row.addWidget(QtWidgets.QLabel("Log window +/- sec"))
        self.window_seconds = QtWidgets.QDoubleSpinBox()
        self.window_seconds.setRange(1.0, 100000.0)
        self.window_seconds.setDecimals(0)
        self.window_seconds.setValue(DEFAULT_EVENT_WINDOW_SECONDS)
        self.window_seconds.valueChanged.connect(lambda _value: self._update_side_panel())
        window_row.addWidget(self.window_seconds)
        side_layout.addLayout(window_row)

        self.event_text = QtWidgets.QPlainTextEdit()
        self.event_text.setReadOnly(True)
        self.event_text.setMaximumBlockCount(500)
        side_layout.addWidget(self.event_text, stretch=1)

        bottom = QtWidgets.QVBoxLayout()
        bottom.setSpacing(6)
        outer.addLayout(bottom)

        self.timeline_plot = pg.PlotWidget()
        self.timeline_plot.setMinimumHeight(120)
        self.timeline_plot.setMaximumHeight(150)
        self.timeline_plot.setBackground("#FFFFFF")
        self.timeline_plot.showGrid(x=True, y=False, alpha=0.18)
        self.timeline_plot.hideAxis("left")
        bottom.addWidget(self.timeline_plot)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        bottom.addLayout(controls)

        self.btn_play = QtWidgets.QPushButton("Pause")
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        controls.addWidget(self.frame_slider, stretch=1)

        controls.addWidget(QtWidgets.QLabel("Goto t"))
        self.goto_edit = QtWidgets.QLineEdit()
        self.goto_edit.setMaximumWidth(90)
        self.goto_edit.returnPressed.connect(self._goto_time)
        controls.addWidget(self.goto_edit)

        controls.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(int(MIN_FRAME_SPEED), int(MAX_FRAME_SPEED))
        self.speed_slider.setValue(int(self.frame_speed))
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        self.speed_slider.setMaximumWidth(190)
        controls.addWidget(self.speed_slider)

        self.speed_label = QtWidgets.QLabel(self._format_speed(self.frame_speed))
        self.speed_label.setMinimumWidth(52)
        controls.addWidget(self.speed_label)

        for speed in (1, 5, 10, 50):
            btn = QtWidgets.QPushButton(f"{speed}x")
            btn.setMaximumWidth(44)
            btn.clicked.connect(lambda _checked=False, value=speed: self._set_speed(value))
            controls.addWidget(btn)

        self.map_plot.scene().sigMouseMoved.connect(self._on_map_mouse_moved)
        self.map_plot.scene().sigMouseClicked.connect(self._on_map_mouse_clicked)

    def _build_background_controls(self, side_layout: QtWidgets.QVBoxLayout):
        group = QtWidgets.QGroupBox("Background")
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self.bg_status_label = QtWidgets.QLabel("No background")
        self.bg_status_label.setWordWrap(True)
        layout.addWidget(self.bg_status_label, 0, 0, 1, 4)

        self.bg_offset_x = self._make_bg_spin(-5000.0, 5000.0, 0.0, 1.0, 1)
        self.bg_offset_y = self._make_bg_spin(-5000.0, 5000.0, 0.0, 1.0, 1)
        self.bg_scale_x = self._make_bg_spin(0.05, 10.0, 1.0, 0.01, 3)
        self.bg_scale_y = self._make_bg_spin(0.05, 10.0, 1.0, 0.01, 3)
        self.bg_opacity = self._make_bg_spin(0.0, 1.0, 0.38, 0.05, 2)

        controls = [
            ("X", self.bg_offset_x),
            ("Y", self.bg_offset_y),
            ("W", self.bg_scale_x),
            ("H", self.bg_scale_y),
            ("A", self.bg_opacity),
        ]
        for index, (label, spin) in enumerate(controls, start=1):
            col = 0 if index % 2 else 2
            row = 1 + (index - 1) // 2
            layout.addWidget(QtWidgets.QLabel(label), row, col)
            layout.addWidget(spin, row, col + 1)

        self.btn_bg_reset = QtWidgets.QPushButton("Reset BG")
        self.btn_bg_reset.clicked.connect(self._reset_background_adjustment)
        layout.addWidget(self.btn_bg_reset, 4, 0, 1, 2)

        self.btn_bg_save = QtWidgets.QPushButton("Save BG")
        self.btn_bg_save.clicked.connect(self._save_background_adjustment)
        layout.addWidget(self.btn_bg_save, 4, 2, 1, 2)

        self.bg_controls = [
            self.bg_offset_x,
            self.bg_offset_y,
            self.bg_scale_x,
            self.bg_scale_y,
            self.bg_opacity,
            self.btn_bg_reset,
            self.btn_bg_save,
        ]
        side_layout.addWidget(group)

    def _make_bg_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(lambda _value: self._on_background_control_changed())
        return spin

    def load_dataset(
        self,
        map_data: dict,
        records: list[dict],
        log_path: Path,
        depot_events: list[dict] | None = None,
        bottleneck_events: list[dict] | None = None,
        metadata: dict | None = None,
        map_path: Path | None = None,
    ):
        self.timer.stop()
        self.map_data = map_data
        self.records = records
        self.log_path = Path(log_path)
        self.map_path = Path(map_path) if map_path is not None else None
        self.metadata = metadata or {}
        self.depot_events = depot_events or []
        self.bottleneck_events = bottleneck_events or []
        self.frames = build_frames(records)
        self.amr_ids = sorted({record["AMR_id"] for record in records if record.get("AMR_id")})
        self.frame_times = [t for t, _items in self.frames]
        self.trace_series, self.trace_counts = build_trace_series(self.frames, self.amr_ids)
        self.position_snapshots = precompute_positions(self.frames, self.amr_ids)
        self.amr_timelines = build_amr_timelines(records, self.amr_ids)
        log_interval = self.metadata.get("motion_log_interval", 0.5)
        try:
            log_interval = float(log_interval)
        except (TypeError, ValueError):
            log_interval = 0.5
        self.motion_log_interval = max(0.001, log_interval)
        self.interpolation_max_gap = max(1.0, log_interval * 2.5)
        self.initial_occupancy = initial_occupancy(map_data)
        self.depot_timeline = build_depot_timeline(
            self.initial_occupancy, self.depot_events
        )
        self.bottleneck_timeline = build_bottleneck_timeline(self.bottleneck_events)
        self.timeline_events = build_timeline_events(
            self.depot_events, self.bottleneck_events
        )
        self.depot_records = {
            record["location_id"]: record for record in iter_depots(map_data)
        }
        self.path_visible = {amr_id: True for amr_id in self.amr_ids}
        self.selected_target = None
        self.selected_event = None
        self.current_frame = 0
        self.current_time = self.frame_times[0] if self.frame_times else 0.0
        self.current_display_positions = {}
        self._last_side_panel_wall_time = 0.0
        self._frame_advance_remainder = 0.0
        self.playing = True
        self.btn_play.setText("Pause")

        self._rebuild_amr_filters()
        self._sync_background_controls()
        self._rebuild_map()
        self._rebuild_timeline()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, len(self.frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.setWindowTitle(f"AMR Replay - {self.log_path.name}")

        if self.frames:
            self.reset_view()
            self._draw_frame(0)
            self.timer.start(self._playback_interval_ms())
        else:
            self.detail_text.setPlainText("Log file contains no AMR records.")

    def _rebuild_amr_filters(self):
        while self.filter_layout.count():
            item = self.filter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.amr_checks: dict[str, QtWidgets.QCheckBox] = {}
        for amr_id in self.amr_ids:
            check = QtWidgets.QCheckBox(amr_id)
            check.setChecked(True)
            check.setStyleSheet(f"color: {AMR_COLORS.get(amr_id, '#333333')};")
            check.stateChanged.connect(
                lambda state, aid=amr_id: self._set_amr_visible(
                    aid, bool(state)
                )
            )
            self.filter_layout.addWidget(check)
            self.amr_checks[amr_id] = check
        self.filter_layout.addStretch(1)

    def _rebuild_map(self):
        self.map_plot.clear()
        self.map_plot.setLabel("bottom", "X")
        self.map_plot.setLabel("left", "Y")
        self.map_plot.showGrid(x=True, y=True, alpha=0.18)
        self.map_plot.setAspectLocked(True)
        self.map_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self._add_background_image()

        self._region_items = []
        region_brush = pg.mkBrush(200, 230, 201, 46)
        region_pen = pg.mkPen(None)
        for region in self.map_data.get("regions", []):
            for cell in region.get("cells", []):
                if len(cell) < 2:
                    continue
                item = QtWidgets.QGraphicsRectItem(float(cell[0]), float(cell[1]), 1.0, 1.0)
                item.setBrush(region_brush)
                item.setPen(region_pen)
                item.setZValue(-20)
                self.map_plot.addItem(item)
                self._region_items.append(item)

        path_nodes = [
            node
            for node in self.map_data.get("amr_path_nodes", [])
            if node.get("x") is not None and node.get("y") is not None
        ]
        xs, ys = path_line_arrays(path_nodes)
        self.path_item = pg.PlotDataItem(
            xs,
            ys,
            pen=pg.mkPen("#D32F2F", width=2),
            connect="finite",
        )
        self.path_item.setZValue(-5)
        self.map_plot.addItem(self.path_item)

        self.depot_groups: dict[str, pg.ScatterPlotItem] = {}
        for group_key, spec in GROUP_SPECS.items():
            scatter = pg.ScatterPlotItem(symbol=spec["symbol"], hoverable=False)
            scatter.setZValue(5)
            self.map_plot.addItem(scatter)
            self.depot_groups[group_key] = scatter

        self.selected_depot_halo = pg.ScatterPlotItem(symbol="o")
        self.selected_depot_halo.setZValue(12)
        self.map_plot.addItem(self.selected_depot_halo)

        self.amr_trace_items: dict[str, pg.PlotDataItem] = {}
        for amr_id in self.amr_ids:
            color = AMR_COLORS.get(amr_id, "#6D4C41")
            line = pg.PlotDataItem([], [], pen=pg.mkPen(color, width=1.6))
            line.setZValue(20)
            self.map_plot.addItem(line)
            self.amr_trace_items[amr_id] = line

        self.amr_body_items: dict[str, QtWidgets.QGraphicsRectItem] = {}
        for amr_id in self.amr_ids:
            body = QtWidgets.QGraphicsRectItem(
                -AMR_BODY_LENGTH_CELLS / 2,
                -AMR_BODY_WIDTH_CELLS / 2,
                AMR_BODY_LENGTH_CELLS,
                AMR_BODY_WIDTH_CELLS,
            )
            body.setZValue(30)
            body.setVisible(False)
            self.map_plot.addItem(body)
            self.amr_body_items[amr_id] = body

        self.amr_carry_badges: dict[str, pg.TextItem] = {}
        for amr_id in self.amr_ids:
            badge = pg.TextItem(
                "",
                color="#263238",
                anchor=(0.5, 0.5),
                fill=pg.mkBrush(255, 255, 255, 230),
                border=pg.mkPen("#263238", width=1.2),
            )
            badge.setZValue(36)
            badge.setVisible(False)
            self.map_plot.addItem(badge)
            self.amr_carry_badges[amr_id] = badge

        self.selected_amr_halo = pg.ScatterPlotItem(symbol="o")
        self.selected_amr_halo.setZValue(29)
        self.map_plot.addItem(self.selected_amr_halo)

        self.amr_labels: dict[str, pg.TextItem] = {}
        for amr_id in self.amr_ids:
            label = pg.TextItem(amr_id, color=AMR_COLORS.get(amr_id, "#333333"), anchor=(0.5, 1.0))
            label.setZValue(35)
            label.setVisible(False)
            self.map_plot.addItem(label)
            self.amr_labels[amr_id] = label

        self.bottleneck_badges: dict[str, pg.TextItem] = {}
        for loc_id in self.bottleneck_timeline:
            record = self.depot_records.get(loc_id)
            if record is None:
                continue
            badge = pg.TextItem(
                "",
                color="#B71C1C",
                anchor=(0.5, 1.0),
                fill=pg.mkBrush(255, 255, 255, 225),
                border=pg.mkPen("#B71C1C", width=1.4),
            )
            badge.setPos(record["_x"], record["_y"] + 0.9)
            badge.setZValue(45)
            badge.setVisible(False)
            self.map_plot.addItem(badge)
            self.bottleneck_badges[loc_id] = badge

    def _add_background_image(self):
        self.background_item = None
        self.background_image_scale = (1.0, 1.0)
        self._set_background_status("No background image configured.")
        spec = _background_spec(self.map_data)
        if spec is None:
            return

        image_path = _background_path(spec, self.map_path)
        if image_path is None:
            self._set_background_status("Background image path is missing.")
            return
        if not image_path.exists():
            self._set_background_status(f"Background image not found: {image_path}")
            return

        pixmap, image_scale, status = _read_background_pixmap(image_path, spec)
        if pixmap is None:
            self._set_background_status(status)
            return
        self.background_image_scale = image_scale

        transform = _background_transform(spec, image_scale=image_scale)
        if transform is None:
            self._set_background_status("Background calibration requires four image/grid point pairs.")
            return

        try:
            opacity = float(spec.get("opacity", 0.38))
        except (TypeError, ValueError):
            opacity = 0.38
        opacity = max(0.0, min(1.0, opacity))

        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)
        item.setTransform(transform)
        item.setOpacity(opacity)
        item.setZValue(-100)
        self.map_plot.addItem(item, ignoreBounds=True)
        self.background_item = item
        self._set_background_status(status)

    def _set_background_status(self, status: str):
        self.background_status = status
        label = getattr(self, "bg_status_label", None)
        if label is not None:
            label.setText(status)

    def _sync_background_controls(self):
        spec = _background_spec(self.map_data)
        enabled = spec is not None
        for control in getattr(self, "bg_controls", []):
            control.setEnabled(enabled)
        if spec is None:
            self._set_background_status("No background image configured.")
            return

        adjustment = _background_adjustment(spec)
        try:
            opacity = float(spec.get("opacity", 0.38))
        except (TypeError, ValueError):
            opacity = 0.38

        self._bg_controls_busy = True
        try:
            self.bg_offset_x.setValue(adjustment["offset_x"])
            self.bg_offset_y.setValue(adjustment["offset_y"])
            self.bg_scale_x.setValue(adjustment["scale_x"])
            self.bg_scale_y.setValue(adjustment["scale_y"])
            self.bg_opacity.setValue(max(0.0, min(1.0, opacity)))
        finally:
            self._bg_controls_busy = False

    def _write_background_controls_to_spec(self) -> dict | None:
        spec = _background_spec(self.map_data)
        if spec is None:
            return None
        spec["opacity"] = float(self.bg_opacity.value())
        spec["adjustment"] = {
            "offset_x": float(self.bg_offset_x.value()),
            "offset_y": float(self.bg_offset_y.value()),
            "scale_x": float(self.bg_scale_x.value()),
            "scale_y": float(self.bg_scale_y.value()),
        }
        return spec

    def _on_background_control_changed(self):
        if self._bg_controls_busy:
            return
        spec = self._write_background_controls_to_spec()
        if spec is None:
            return

        if self.background_item is None:
            self._add_background_image()
            return

        transform = _background_transform(
            spec,
            image_scale=self.background_image_scale,
        )
        if transform is None:
            self._set_background_status("Background calibration requires four image/grid point pairs.")
            return

        self.background_item.setTransform(transform)
        self.background_item.setOpacity(float(spec.get("opacity", 0.38)))
        self.map_plot.scene().update()

    def _reset_background_adjustment(self):
        self._bg_controls_busy = True
        try:
            self.bg_offset_x.setValue(0.0)
            self.bg_offset_y.setValue(0.0)
            self.bg_scale_x.setValue(1.0)
            self.bg_scale_y.setValue(1.0)
            self.bg_opacity.setValue(0.38)
        finally:
            self._bg_controls_busy = False
        self._on_background_control_changed()

    def _save_background_adjustment(self):
        spec = self._write_background_controls_to_spec()
        if spec is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Save BG",
                "No background image is configured for this map.",
            )
            return
        if self.map_path is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Save BG",
                "Map path is not available, so the background adjustment cannot be saved.",
            )
            return

        try:
            data = load_json(self.map_path)
            metadata = data.setdefault("metadata", {})
            file_spec = metadata.get("background_image")
            if not isinstance(file_spec, dict):
                file_spec = {}
                metadata["background_image"] = file_spec
            file_spec["opacity"] = spec["opacity"]
            file_spec["adjustment"] = dict(spec["adjustment"])
            with self.map_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save BG Failed", str(exc))
            return

        QtWidgets.QMessageBox.information(
            self,
            "Save BG",
            f"Background adjustment saved to:\n{self.map_path}",
        )

    def _rebuild_timeline(self):
        self.timeline_plot.clear()
        self.timeline_plot.hideAxis("left")
        self.timeline_plot.setLabel("bottom", "simulation time t")
        self.timeline_plot.setYRange(-0.6, 1.6)
        self.timeline_plot.showGrid(x=True, y=False, alpha=0.18)
        self.timeline_marker_item = pg.ScatterPlotItem(hoverable=False)
        self.timeline_marker_item.setZValue(10)
        self.timeline_marker_item.sigClicked.connect(self._on_timeline_marker_clicked)
        self.timeline_plot.addItem(self.timeline_marker_item)
        self.timeline_cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#263238", width=2))
        self.timeline_cursor.setZValue(20)
        self.timeline_plot.addItem(self.timeline_cursor)
        self.timeline_plot.getViewBox().sigXRangeChanged.connect(self._refresh_timeline_markers)

        if self.frame_times:
            self.timeline_plot.setXRange(self.frame_times[0], self.frame_times[-1], padding=0.02)
        self._refresh_timeline_markers()

    def _depot_spot(self, loc_id: str, record: dict, time_val: float):
        group_key = record["_group_key"]
        spec = GROUP_SPECS[group_key]
        color = spec["color"]
        entry = self.depot_timeline.get(loc_id)
        occupied = occupancy_at(entry, time_val) if entry is not None else 0
        bottleneck_entry = self.bottleneck_timeline.get(loc_id)
        current, total = bottleneck_at(bottleneck_entry, time_val) if bottleneck_entry else (0, 0)
        is_bottleneck = current > 0 or total > 0
        is_selected = self.selected_target == ("depot", loc_id)

        brush = pg.mkBrush(color if occupied else "#FFFFFF")
        if is_bottleneck:
            pen = pg.mkPen("#B71C1C", width=3.0)
        elif is_selected:
            pen = pg.mkPen("#FBC02D", width=3.0)
        else:
            pen = pg.mkPen(color, width=1.2)
        return {
            "pos": (record["_x"], record["_y"]),
            "data": loc_id,
            "brush": brush,
            "pen": pen,
            "size": spec["size"],
        }

    def _draw_depots(self, time_val: float):
        grouped: dict[str, list] = {key: [] for key in GROUP_SPECS}
        halo_spots = []
        for loc_id, record in self.depot_records.items():
            grouped[record["_group_key"]].append(self._depot_spot(loc_id, record, time_val))
            if self.selected_target == ("depot", loc_id):
                halo_spots.append(
                    {
                        "pos": (record["_x"], record["_y"]),
                        "brush": pg.mkBrush(255, 235, 59, 35),
                        "pen": pg.mkPen("#FBC02D", width=3.0),
                        "size": 24,
                    }
                )

        for group_key, scatter in self.depot_groups.items():
            scatter.setData(grouped[group_key])
        self.selected_depot_halo.setData(halo_spots)

    def _draw_amrs(
        self,
        fi: int,
        current_positions: dict[str, tuple[float, float, dict, float]] | None = None,
    ):
        if current_positions is None:
            snapshot = self.position_snapshots[fi] if fi < len(self.position_snapshots) else {}
            current_positions = {
                amr_id: (
                    x,
                    y,
                    item,
                    heading_degrees(
                        self.trace_series[amr_id][0],
                        self.trace_series[amr_id][1],
                        self.trace_counts[amr_id][fi],
                    ),
                )
                for amr_id, (x, y, item) in snapshot.items()
            }
        selected_halo = []
        full_trace = self.full_trace_check.isChecked()

        for amr_id in self.amr_ids:
            line = self.amr_trace_items[amr_id]
            label = self.amr_labels[amr_id]
            body = self.amr_body_items[amr_id]
            carry_badge = self.amr_carry_badges[amr_id]
            visible = self.path_visible.get(amr_id, True)

            if not visible or amr_id not in current_positions:
                line.setData([], [])
                label.setVisible(False)
                body.setVisible(False)
                carry_badge.setVisible(False)
                continue

            x, y, item, heading = current_positions[amr_id]
            color = AMR_COLORS.get(amr_id, "#6D4C41")
            selected = self.selected_target == ("amr", amr_id)
            status = item.get("amr_status")
            carrying_type = carrying_type_from_record(item)
            if status == "UNDISPATCHED":
                brush = pg.mkBrush(255, 255, 255, 0)
                pen = pg.mkPen(color, width=2.4)
            else:
                brush = pg.mkBrush(color)
                pen = pg.mkPen("#FFFFFF", width=1.5)

            xs, ys = self.trace_series[amr_id]
            count = self.trace_counts[amr_id][fi]
            body.setBrush(brush)
            pen_color = CARRYING_BADGE_COLORS.get(carrying_type, pen.color())
            pen_width = 4.2 if selected else (3.0 if carrying_type else pen.widthF())
            body.setPen(pg.mkPen(pen_color, width=pen_width))
            body.setPos(float(x), float(y))
            body.setRotation(heading)
            body.setVisible(True)
            carry_badge.setVisible(False)

            if selected:
                selected_halo.append(
                    {
                        "pos": (x, y),
                        "brush": pg.mkBrush(255, 235, 59, 35),
                        "pen": pg.mkPen("#FBC02D", width=3.0),
                        "size": 26,
                    }
                )

            label.setPos(x, y + AMR_BODY_WIDTH_CELLS / 2 + 0.45)
            label.setVisible(True)

            if full_trace:
                start = 0
            else:
                start = max(0, count - self.trace_max_points) if self.trace_max_points else 0
            line.setData(xs[start:count], ys[start:count])
            line.setPen(pg.mkPen(color, width=3.2 if selected else 1.6))

        self.selected_amr_halo.setData(selected_halo)

    def _draw_bottlenecks(self, time_val: float):
        for loc_id, badge in self.bottleneck_badges.items():
            entry = self.bottleneck_timeline.get(loc_id)
            if entry is None:
                badge.setVisible(False)
                continue
            current, total = bottleneck_at(entry, time_val)
            if current == 0 and total == 0:
                badge.setVisible(False)
                continue
            badge.setText(f"({current}/{total})")
            badge.setVisible(True)

    def _draw_frame(self, fi: int):
        if not self.frames:
            return
        fi = max(0, min(fi, len(self.frames) - 1))
        self._draw_time(self.frame_times[fi], fi)

    def _draw_time(self, time_val: float, fi: int | None = None):
        if not self.frames:
            return
        if fi is None:
            fi = bisect.bisect_right(self.frame_times, time_val) - 1
        fi = max(0, min(fi, len(self.frames) - 1))
        time_val = max(self.frame_times[0], min(float(time_val), self.frame_times[-1]))
        self.current_frame = fi
        self.current_time = time_val
        self.current_display_positions = interpolated_amr_positions(
            time_val,
            self.amr_timelines,
            self.interpolation_max_gap,
        )
        self._draw_depots(time_val)
        self._draw_amrs(fi, self.current_display_positions)
        self._draw_bottlenecks(time_val)
        self.timeline_cursor.setValue(time_val)
        self.time_label.setText(
            f"t = {time_val:.2f}    frame {fi + 1}/{len(self.frames)}    {self._format_speed(self.frame_speed)}"
        )
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(fi)
        self.frame_slider.blockSignals(False)
        self._maybe_update_side_panel()

    def _timer_tick(self):
        if not self.frames:
            return
        if not self.playing:
            return
        playback_interval = self._playback_interval_ms()
        sim_seconds_per_wall_second = (
            self.motion_log_interval
            * 1000.0
            / max(1, self.interval_ms)
        )
        self.current_time += (
            min(
                self.motion_log_interval,
                playback_interval
                / 1000.0
                * self.frame_speed
                * sim_seconds_per_wall_second,
            )
        )
        if self.current_time > self.frame_times[-1]:
            self.current_time = self.frame_times[0]
        self._draw_time(self.current_time)

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

    def _playback_interval_ms(self) -> int:
        return max(
            MIN_PLAYBACK_INTERVAL_MS,
            min(
                33,
                int(round(self.interval_ms / max(MIN_FRAME_SPEED, self.frame_speed))),
            ),
        )

    def _format_speed(self, value: float) -> str:
        return f"{float(value):.2f}x"

    def _toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")

    def _pause(self):
        self.playing = False
        self.btn_play.setText("Play")

    def _seek_to_frame(self, fi: int, pause: bool = False):
        if not self.frames:
            return
        if pause:
            self._pause()
        self._frame_advance_remainder = 0.0
        self._last_side_panel_wall_time = 0.0
        self._draw_frame(max(0, min(fi, len(self.frames) - 1)))

    def _seek_to_time(self, time_val: float, pause: bool = False):
        if not self.frames:
            return
        if pause:
            self._pause()
        self._frame_advance_remainder = 0.0
        self._last_side_panel_wall_time = 0.0
        self._draw_time(time_val)

    def _on_frame_slider_changed(self, value: int):
        self._seek_to_frame(value, pause=False)

    def _goto_time(self):
        text = self.goto_edit.text().strip()
        if not text:
            return
        try:
            t = float(text)
        except ValueError:
            return
        self._seek_to_time(t, pause=True)

    def _on_speed_slider_changed(self, value: int):
        self._set_speed(float(value))

    def _set_speed(self, value: float):
        self.frame_speed = clamp_speed(value)
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(int(round(self.frame_speed)))
        self.speed_slider.blockSignals(False)
        self.speed_label.setText(self._format_speed(self.frame_speed))
        self._frame_advance_remainder = 0.0
        self.timer.setInterval(self._playback_interval_ms())
        self._draw_time(self.current_time)

    def _set_amr_visible(self, amr_id: str, visible: bool):
        self.path_visible[amr_id] = visible
        if self.selected_target == ("amr", amr_id) and not visible:
            self.selected_target = None
            self._last_side_panel_wall_time = 0.0
        self._draw_time(self.current_time)

    def reset_view(self):
        x_min, x_max, y_min, y_max = visible_bounds(self.map_data)
        self.map_plot.setXRange(x_min, x_max, padding=0.02)
        self.map_plot.setYRange(y_min, y_max, padding=0.02)

    def _hit_radius(self) -> float:
        try:
            (x_min, x_max), (y_min, y_max) = self.map_plot.viewRange()
            px_w = max(1, self.map_plot.width())
            px_h = max(1, self.map_plot.height())
            return max((x_max - x_min) / px_w * 16, (y_max - y_min) / px_h * 16, 0.45)
        except Exception:
            return 0.6

    def _nearest_amr(self, point: QtCore.QPointF):
        if not self.frames:
            return None
        current_positions = self.current_display_positions or {
            amr_id: (x, y, item, 0.0)
            for amr_id, (x, y, item) in self.position_snapshots[self.current_frame].items()
        }
        radius = max(self._hit_radius(), AMR_BODY_LENGTH_CELLS / 2)
        best = None
        for amr_id, (x, y, item, _heading) in current_positions.items():
            if not self.path_visible.get(amr_id, True):
                continue
            dist = math.hypot(point.x() - x, point.y() - y)
            if dist <= radius and (best is None or dist < best[0]):
                best = (dist, amr_id, item)
        return best

    def _nearest_depot(self, point: QtCore.QPointF):
        radius = self._hit_radius()
        best = None
        for loc_id, record in self.depot_records.items():
            dist = math.hypot(point.x() - record["_x"], point.y() - record["_y"])
            if dist <= radius and (best is None or dist < best[0]):
                best = (dist, loc_id, record)
        return best

    def _on_map_mouse_moved(self, scene_pos: QtCore.QPointF):
        view_box = self.map_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            self._last_tooltip_key = None
            QtWidgets.QToolTip.hideText()
            return
        point = view_box.mapSceneToView(scene_pos)
        amr = self._nearest_amr(point)
        depot = self._nearest_depot(point)
        target = None
        if amr and depot:
            target = ("amr", amr) if amr[0] <= depot[0] else ("depot", depot)
        elif amr:
            target = ("amr", amr)
        elif depot:
            target = ("depot", depot)

        if target is None:
            self._last_tooltip_key = None
            QtWidgets.QToolTip.hideText()
            return

        kind, payload = target
        key = (kind, payload[1])
        if self._last_tooltip_key == key:
            return
        self._last_tooltip_key = key
        if kind == "amr":
            _dist, amr_id, item = payload
            text = self._format_amr_tooltip(amr_id, item)
        else:
            _dist, loc_id, record = payload
            text = self._format_depot_tooltip(loc_id, record)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self.map_plot)

    def _on_map_mouse_clicked(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        view_box = self.map_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view_box.mapSceneToView(event.scenePos())
        amr = self._nearest_amr(point)
        depot = self._nearest_depot(point)
        if amr and (not depot or amr[0] <= depot[0]):
            self.selected_target = ("amr", amr[1])
            self.selected_event = None
            event.accept()
        elif depot:
            self.selected_target = ("depot", depot[1])
            self.selected_event = None
            event.accept()
        else:
            return
        self._last_side_panel_wall_time = 0.0
        self._draw_time(self.current_time)

    def _format_amr_tooltip(self, amr_id: str, item: dict) -> str:
        return "\n".join(
            [
                f"AMR: {amr_id}",
                f"status: {item.get('amr_status', '')}",
                f"task_step: {item.get('task_step', '')}",
                f"result: {item.get('result_status', '')}",
            ]
        )

    def _format_depot_tooltip(self, loc_id: str, record: dict) -> str:
        time_val = self.current_time if self.frames else 0.0
        occ_entry = self.depot_timeline.get(loc_id)
        occ = occupancy_at(occ_entry, time_val) if occ_entry else 0
        bn_entry = self.bottleneck_timeline.get(loc_id)
        current, total = bottleneck_at(bn_entry, time_val) if bn_entry else (0, 0)
        return "\n".join(
            [
                f"Depot: {loc_id}",
                f"type: {GROUP_SPECS[record['_group_key']]['name']}",
                f"rack_type: {record.get('rack_type', '')}",
                f"occupied: {occ}",
                f"bottleneck: {current}/{total}",
            ]
        )

    def _refresh_timeline_markers(self, *_args):
        if not hasattr(self, "timeline_marker_item"):
            return
        if not self.timeline_events:
            self.timeline_marker_item.setData([])
            return

        x_range = self.timeline_plot.viewRange()[0]
        x_min, x_max = min(x_range), max(x_range)
        span = max(1.0, x_max - x_min)
        visible = [
            event
            for event in self.timeline_events
            if x_min - span * 0.05 <= event["time"] <= x_max + span * 0.05
        ]
        aggregate = span > 600.0 and len(visible) > 180
        spots = []
        if aggregate:
            bin_count = min(120, max(24, int(self.timeline_plot.width() / 12)))
            bin_width = span / bin_count
            buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
            for event in visible:
                idx = int((event["time"] - x_min) / bin_width)
                idx = max(0, min(bin_count - 1, idx))
                buckets[(idx, event["kind"])].append(event)
            for (idx, kind), bucket in buckets.items():
                t0 = x_min + idx * bin_width
                t1 = t0 + bin_width
                time_val = sum(event["time"] for event in bucket) / len(bucket)
                color = "#B71C1C" if kind == "bottleneck" else "#546E7A"
                y = 1.0 if kind == "bottleneck" else 0.0
                spots.append(
                    {
                        "pos": (time_val, y),
                        "data": {
                            "aggregate": True,
                            "kind": kind,
                            "count": len(bucket),
                            "time_start": t0,
                            "time_end": t1,
                            "events": bucket,
                        },
                        "brush": pg.mkBrush(color),
                        "pen": pg.mkPen("#FFFFFF", width=1.0),
                        "size": min(22, 7 + math.sqrt(len(bucket)) * 2.0),
                    }
                )
        else:
            for event in visible:
                kind = event["kind"]
                y = 1.0 if kind == "bottleneck" else 0.0
                color = "#B71C1C" if kind == "bottleneck" else "#607D8B"
                spots.append(
                    {
                        "pos": (event["time"], y),
                        "data": {"aggregate": False, "event": event},
                        "brush": pg.mkBrush(color),
                        "pen": pg.mkPen("#FFFFFF", width=1.0),
                        "size": 9 if kind == "bottleneck" else 7,
                    }
                )
        self.timeline_marker_item.setData(spots)

    def _on_timeline_marker_clicked(self, _scatter, points):
        if not points:
            return
        data = points[0].data()
        if data.get("aggregate"):
            self.timeline_plot.setXRange(data["time_start"], data["time_end"], padding=0.15)
            return
        event = data.get("event")
        if not event:
            return
        self.selected_event = event
        if event.get("AMR_id"):
            self.selected_target = ("amr", event["AMR_id"])
        elif event.get("location_id"):
            self.selected_target = ("depot", event["location_id"])
        self._last_side_panel_wall_time = 0.0
        self._seek_to_time(float(event["time"]), pause=True)

    def _is_event_related(self, event: dict) -> bool:
        if self.selected_target is None:
            return False
        kind, ident = self.selected_target
        if kind == "amr":
            return event.get("AMR_id") == ident or event.get("raw", {}).get("AMR_id") == ident
        if kind == "depot":
            return event.get("location_id") == ident
        return False

    def _maybe_update_side_panel(self):
        now = time.monotonic()
        if (
            self._last_side_panel_wall_time
            and now - self._last_side_panel_wall_time < 0.15
        ):
            return
        self._last_side_panel_wall_time = now
        self._update_side_panel()

    def _update_side_panel(self):
        if not self.frames:
            return
        time_val = self.current_time
        self.detail_text.setPlainText(self._selection_text(time_val))
        self.event_text.setPlainText(self._event_log_text(time_val))

    def _selection_text(self, time_val: float) -> str:
        lines = []
        if self.selected_event:
            lines.extend(["Pinned event", self._format_event(self.selected_event), ""])

        if self.selected_target is None:
            lines.append("No selection.")
            return "\n".join(lines)

        kind, ident = self.selected_target
        if kind == "amr":
            item = (self.current_display_positions or {}).get(ident)
            if item is None:
                lines.append(f"AMR {ident} is not visible at this time.")
                return "\n".join(lines)
            x, y, record, _heading = item
            lines.extend(
                [
                    f"AMR: {ident}",
                    f"x/y: {x - 0.5:.1f}, {y - 0.5:.1f}",
                    f"status: {record.get('amr_status', '')}",
                    f"task_step: {record.get('task_step', '')}",
                    f"carrying_type: {record.get('carrying_type') or carrying_type_from_record(record) or '미적재'}",
                    f"load_type: {record.get('load_type', '')}",
                    f"result_status: {record.get('result_status', '')}",
                    f"speed: {record.get('speed', '')}",
                    f"move_num: {record.get('move_num', '')}",
                ]
            )
            return "\n".join(lines)

        if kind == "depot":
            record = self.depot_records.get(ident)
            if record is None:
                lines.append(f"Depot {ident} not found in map.")
                return "\n".join(lines)
            occ_entry = self.depot_timeline.get(ident)
            occ = occupancy_at(occ_entry, time_val) if occ_entry else 0
            bn_entry = self.bottleneck_timeline.get(ident)
            current, total = bottleneck_at(bn_entry, time_val) if bn_entry else (0, 0)
            lines.extend(
                [
                    f"Depot: {ident}",
                    f"type: {GROUP_SPECS[record['_group_key']]['name']}",
                    f"rack_type: {record.get('rack_type', '')}",
                    f"occupied: {occ}",
                    f"bottleneck current/total: {current}/{total}",
                    "",
                    "Bottleneck history",
                ]
            )
            history = [
                event
                for event in self.bottleneck_events
                if event.get("location_id") == ident
            ][-12:]
            if not history:
                lines.append("(none)")
            else:
                for event in history:
                    lines.append(
                        f"t={float(event.get('time') or 0):.1f} "
                        f"{event.get('event_type', '')} "
                        f"{event.get('current', 0)}/{event.get('total', 0)} "
                        f"AMR={event.get('AMR_id', '')}"
                    )
            return "\n".join(lines)

        return "No selection."

    def _event_log_text(self, time_val: float) -> str:
        window = float(self.window_seconds.value())
        events = [
            event
            for event in self.timeline_events
            if time_val - window <= event["time"] <= time_val + window
        ]
        events.sort(key=lambda event: (not self._is_event_related(event), abs(event["time"] - time_val), event["time"]))
        lines = [f"Events near t={time_val:.1f} (+/- {window:.0f}s)"]
        if not events:
            lines.append("(none)")
            return "\n".join(lines)
        for event in events[:80]:
            prefix = "*" if self._is_event_related(event) else " "
            lines.append(prefix + " " + self._format_event(event))
        return "\n".join(lines)

    def _format_event(self, event: dict) -> str:
        raw = event.get("raw", {})
        if event["kind"] == "bottleneck":
            return (
                f"[B] t={event['time']:.1f} loc={event.get('location_id', '')} "
                f"{raw.get('event_type', '')} "
                f"{raw.get('current', 0)}/{raw.get('total', 0)} "
                f"AMR={raw.get('AMR_id', '')} task={raw.get('task_id', '')} "
                f"part={raw.get('part', '')}"
            )
        return (
            f"[D] t={event['time']:.1f} loc={event.get('location_id', '')} "
            f"occupied={raw.get('occupied', '')} source={raw.get('source', '')}"
        )

    def _on_open_log(self):
        path_text, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open AMR motion log",
            str(LOG_DIR),
            "AMR logs (amr_motion_*.json *.json);;JSON files (*.json);;All files (*)",
        )
        if not path_text:
            return
        try:
            map_data, records, depot_events, bottleneck_events, metadata, map_path = load_replay_payload(Path(path_text))
        except Exception as exc:  # pragma: no cover - user-facing dialog path.
            QtWidgets.QMessageBox.critical(self, "Open Log Failed", str(exc))
            return
        if not records:
            QtWidgets.QMessageBox.warning(self, "Open Log", "Selected log contains no AMR records.")
            return
        self.load_dataset(
            map_data,
            records,
            Path(path_text),
            depot_events=depot_events,
            bottleneck_events=bottleneck_events,
            metadata=metadata,
            map_path=map_path,
        )

    def _show_info(self):
        lines = [
            f"Log: {self.log_path}",
            f"Map: {self.map_path or '(provided by caller)'}",
            self.background_status,
            f"Records: {len(self.records)}",
            f"Frames: {len(self.frames)}",
            f"Depot events: {len(self.depot_events)}",
            f"Bottleneck events: {len(self.bottleneck_events)}",
        ]
        if self.metadata:
            lines.append("")
            lines.append("Metadata")
            for key in sorted(self.metadata):
                lines.append(f"{key}: {self.metadata[key]}")
        QtWidgets.QMessageBox.information(self, "Replay Info", "\n".join(lines))


def run_replayer(
    map_data: dict,
    records: list[dict],
    log_path: Path,
    depot_events: list[dict] | None = None,
    bottleneck_events: list[dict] | None = None,
    metadata: dict | None = None,
    map_path: Path | None = None,
    interval_ms: int = 100,
    initial_speed: float = 1.0,
    trace_max_points: int = DEFAULT_TRACE_MAX_POINTS,
):
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    window = ReplayWindow(
        map_data,
        records,
        log_path,
        depot_events=depot_events,
        bottleneck_events=bottleneck_events,
        metadata=metadata,
        map_path=map_path,
        interval_ms=interval_ms,
        initial_speed=initial_speed,
        trace_max_points=trace_max_points,
    )
    window.show()
    if owns_app:
        return app.exec()
    return window


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Qt visualizer for AMR movement logs.")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to amr_motion_*.json. Defaults to latest in simulation_log/.",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Path to map JSON. Defaults to log metadata or selected config map.",
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

    map_path = Path(args.map) if args.map else None
    map_data, records, depot_events, bottleneck_events, metadata, resolved_map_path = load_replay_payload(
        log_path, map_path
    )
    if not records:
        print("Log file contains no records.")
        sys.exit(1)

    print(f"Map : {resolved_map_path}")
    print(f"Log : {log_path}")
    print(
        f"Loaded {len(records)} AMR records / {len(build_frames(records))} frames"
        f" / {len(depot_events)} depot events"
        f" / {len(bottleneck_events)} bottleneck events."
    )
    sys.exit(
        run_replayer(
            map_data,
            records,
            log_path,
            depot_events=depot_events,
            bottleneck_events=bottleneck_events,
            metadata=metadata,
            map_path=resolved_map_path,
            interval_ms=args.interval_ms,
            initial_speed=args.speed,
            trace_max_points=args.trace_window,
        )
    )


if __name__ == "__main__":
    main()
