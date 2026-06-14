"""Run simulation, collect AMR movement logs, save to simulation_log/, then visualize.

Usage:
    python run_and_visualize.py
    python run_and_visualize.py --product oven
    python run_and_visualize.py --max-time 500
    python run_and_visualize.py --motion-log-interval 0.2
    python run_and_visualize.py --trace-window 300
    python run_and_visualize.py --speed 10 --interval-ms 100
    python run_and_visualize.py --no-show
    python run_and_visualize.py --show-sim-log
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime_product import add_product_argument, choose_product, configure_product_env

LOG_DIR = ROOT_DIR / "simulation_log"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_location_index(map_data: dict) -> dict:
    index = {}
    for key in ("wh_locations", "ml_nodes", "ps_nodes"):
        for record in map_data.get(key, []):
            loc_id = record.get("location_id")
            if loc_id is not None:
                index[loc_id] = record
    return index


def _point_from_location(value, index: dict):
    if value is None:
        return None
    if isinstance(value, str):
        value = index.get(value)
    if isinstance(value, dict):
        coord = value.get("coordinates")
        if isinstance(coord, dict):
            return coord.get("x"), coord.get("y")
        if value.get("x") is not None and value.get("y") is not None:
            return value.get("x"), value.get("y")
    return None


def _find_model(engine, model_id: str):
    for model in engine.models:
        if getattr(model, "getModelID", lambda: None)() == model_id:
            return model
    return None


def _occupancy_snapshot(map_data: dict) -> dict:
    """Return {location_id: occupied} across wh_locations / ml_nodes / ps_nodes."""
    snap = {}
    for key in ("wh_locations", "ml_nodes", "ps_nodes"):
        for record in map_data.get(key, []):
            loc_id = record.get("location_id")
            if loc_id is None:
                continue
            snap[loc_id] = 1 if record.get("occupied") else 0
    return snap


def _location_id_from_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("location_id")
    return getattr(value, "location_id", None)


def _event_value(event, key):
    attr = "from_" if key == "from" else key
    if isinstance(event, dict):
        if key == "from":
            return event.get("from", event.get("from_"))
        return event.get(key)
    return getattr(event, attr, None)


def _event_items(event):
    if isinstance(event, list):
        return event
    return [event]


def _bottleneck_event_from_item(item):
    if not _event_value(item, "bottleneck_event"):
        return None

    loc_id = (
        _event_value(item, "location_id")
        or _location_id_from_value(_event_value(item, "from"))
        or _location_id_from_value(_event_value(item, "to"))
    )
    if loc_id is None:
        return None

    def _int_event_value(key):
        try:
            return int(float(_event_value(item, key) or 0))
        except (TypeError, ValueError):
            return 0

    return {
        "time": _event_value(item, "timestamp"),
        "location_id": loc_id,
        "depot_type": (
            _event_value(item, "bottleneck_depot_type")
            or _event_value(item, "type")
        ),
        "event_type": _event_value(item, "bottleneck_event_type"),
        "current": _int_event_value("bottleneck_current"),
        "total": _int_event_value("bottleneck_total"),
        "delta_current": _int_event_value("bottleneck_delta_current"),
        "delta_total": _int_event_value("bottleneck_delta_total"),
        "AMR_id": _event_value(item, "AMR_id"),
        "task_id": _event_value(item, "task_id"),
        "product": _event_value(item, "product"),
        "part": _event_value(item, "part"),
    }


def run_and_collect(
    max_time: float,
    map_data: dict,
    quiet: bool = True,
    motion_log_interval: float | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    from SimulationEngine.SimulationEngine import SimulationEngine
    from SimulationEngine.Utility.Configurator import Configurator
    from modeling.outmost import Outmost
    from config import AMR_LIST

    location_index = _build_location_index(map_data)
    records: list[dict] = []
    depot_events: list[dict] = []
    bottleneck_events: list[dict] = []

    def append_motion_record(
        event,
        *,
        task_step=None,
        result_status=None,
        speed=None,
        time_offset: float = 0.0,
    ):
        point = _point_from_location(
            getattr(event, "current_location", None), location_index
        )
        if point is None:
            return

        event_time = getattr(event, "timestamp", None)
        if time_offset and event_time is not None:
            try:
                event_time = float(event_time) + time_offset
            except (TypeError, ValueError):
                pass

        x, y = point
        records.append({
            "time": event_time,
            "AMR_id": getattr(event, "AMR_id", None),
            "x": x,
            "y": y,
            "task_step": (
                task_step
                if task_step is not None
                else getattr(event, "task_step", None)
            ),
            "move_num": getattr(event, "move_num", None),
            "amr_status": getattr(event, "amr_status", None),
            "result_status": (
                result_status
                if result_status is not None
                else getattr(event, "result_status", None)
            ),
            "speed": speed if speed is not None else getattr(event, "speed", None),
            "travel_distance": getattr(event, "travel_distance", None),
            "load_type": getattr(event, "load_type", None),
            "carrying_type": getattr(event, "carrying_type", None),
            "idle_type": getattr(event, "idle_type", None),
        })

    cfg = Configurator()
    amr_configs = [dict(item) for item in AMR_LIST]
    if motion_log_interval is not None:
        interval = max(0.01, float(motion_log_interval))
        for amr_config in amr_configs:
            amr_config["physical_log_interval"] = interval
    cfg.addConfiguration("AMR_LIST", amr_configs)

    outmost = Outmost(cfg)
    engine = SimulationEngine()
    engine.setOutmostModel(outmost)

    analyzer = _find_model(engine, "Analyzer")
    if analyzer is None:
        raise RuntimeError("Analyzer model not found in engine.")

    original = analyzer.funcExternalTransition

    def record_transition(port, event):
        if port == "physical_result_from_amr":
            append_motion_record(event)
        elif port == "work_info_from_mm":
            for item in _event_items(event):
                bottleneck_event = _bottleneck_event_from_item(item)
                if bottleneck_event is not None:
                    bottleneck_events.append(bottleneck_event)
        original(port, event)

    analyzer.funcExternalTransition = record_transition

    tm = _find_model(engine, "Transportation manager")
    if tm is not None:
        orig_tm_ext = tm.funcExternalTransition

        def tm_ext(port, event):
            if (
                port == "return_completed_from_amr"
                and getattr(event, "idle_type", None) == "STATION_IDLE"
            ):
                append_motion_record(
                    event,
                    task_step="STATION_IDLE",
                    result_status="SUCCESS",
                    speed=0.0,
                    time_offset=1e-9,
                )
            orig_tm_ext(port, event)

        tm.funcExternalTransition = tm_ext

    # ------------------------------------------------------------------
    # Depot occupancy capture
    # ------------------------------------------------------------------
    mm = _find_model(engine, "Manufacturing manager")
    worker = _find_model(engine, "Worker")

    mm_snapshot = {"data": _occupancy_snapshot(mm.map_data)} if mm is not None else None

    def _emit_diff(time_val):
        if mm is None or mm_snapshot is None:
            return
        new_snap = _occupancy_snapshot(mm.map_data)
        prev = mm_snapshot["data"]
        for loc_id, occ in new_snap.items():
            if prev.get(loc_id) != occ:
                depot_events.append({
                    "time": time_val,
                    "location_id": loc_id,
                    "occupied": occ,
                    "source": "MM",
                })
        mm_snapshot["data"] = new_snap

    if mm is not None:
        orig_mm_ext = mm.funcExternalTransition
        orig_mm_int = mm.funcInternalTransition

        def mm_ext(port, event):
            orig_mm_ext(port, event)
            _emit_diff(mm.getTime())

        def mm_int():
            orig_mm_int()
            _emit_diff(mm.getTime())

        mm.funcExternalTransition = mm_ext
        mm.funcInternalTransition = mm_int

    if worker is not None:
        orig_w_out = worker.funcOutput

        def worker_out():
            try:
                state = worker.getStateValue("state")
            except Exception:
                state = None

            orig_w_out()

            request_messages = []
            if state == "REQUEST" and hasattr(worker, "pending_recovery_requests"):
                if worker.pending_recovery_requests:
                    request_messages = [worker.pending_recovery_requests[0]]
            elif state == "WORK" and hasattr(worker, "active_jobs"):
                completed_jobs = worker._completed_jobs()
                request_messages = [
                    job.get("request_message")
                    for job in completed_jobs
                ]
            elif state == "WORK" and getattr(worker, "active_job", None) is not None:
                request_messages = [worker.active_job.get("request_message")]

            for req in request_messages:
                if req is not None:
                    loc_id = _location_id_from_value(getattr(req, "from_", None))
                    if loc_id is not None:
                        depot_events.append({
                            "time": worker.getTime(),
                            "location_id": loc_id,
                            "occupied": 1,
                            "source": "Worker",
                        })

        worker.funcOutput = worker_out

    run_kwargs = dict(
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

    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                engine.run(**run_kwargs)
    else:
        engine.run(**run_kwargs)

    records.sort(key=lambda r: (float(r["time"] or 0), r["AMR_id"] or ""))
    depot_events.sort(key=lambda d: (float(d["time"] or 0), str(d.get("location_id", ""))))
    bottleneck_events.sort(
        key=lambda d: (float(d["time"] or 0), str(d.get("location_id", "")))
    )
    return records, depot_events, bottleneck_events


def save_records(
    records: list[dict],
    depot_events: list[dict],
    bottleneck_events: list[dict],
    path: Path,
    metadata: dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "amr_motion": records,
        "depot_events": depot_events,
        "bottleneck_events": bottleneck_events,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simulation and visualize AMR movement."
    )
    add_product_argument(parser)
    parser.add_argument(
        "--max-time", type=float, default=None,
        help="Simulation max time. Defaults to the selected product config.",
    )
    parser.add_argument(
        "--interval-ms", type=int, default=100,
        help="Animation frame interval in ms (default: 100).",
    )
    parser.add_argument(
        "--speed", "--frame-speed", dest="speed", type=float, default=1.0,
        help="Initial visualizer frame speed multiplier, clamped to 1.00-100.00.",
    )
    parser.add_argument(
        "--motion-log-interval",
        type=float,
        default=0.5,
        help="AMR motion sample interval in simulation seconds (default: 0.5). Use a larger value for lighter replay logs.",
    )
    parser.add_argument(
        "--trace-window",
        type=int,
        default=600,
        help="Number of recent points to draw per AMR trace. Use 0 for full traces.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Collect and save log without opening the visualizer.",
    )
    parser.add_argument(
        "--show-sim-log", action="store_true",
        help="Print simulation output while collecting data.",
    )
    parser.add_argument(
        "--ui",
        choices=("ask", "qt", "mpl", "matplotlib", "none"),
        default="ask",
        help=(
            "Visualizer UI to open after collection. "
            "Use qt for PySide6/pyqtgraph, mpl for the existing Matplotlib UI, "
            "none to skip, or ask to choose interactively."
        ),
    )
    return parser.parse_args()


def choose_ui_interactively() -> str:
    if not sys.stdin.isatty():
        return "qt"

    print()
    print("시각화 UI를 선택하세요.")
    print("  1) Qt / pyqtgraph replay (fast)")
    print("  2) Matplotlib replay (legacy)")
    print("  3) 열지 않음")
    while True:
        choice = input("선택 [1/2/3, 기본=1]: ").strip().lower()
        if choice in ("", "1", "q", "qt"):
            return "qt"
        if choice in ("2", "m", "mpl", "matplotlib"):
            return "mpl"
        if choice in ("3", "n", "none", "no"):
            return "none"
        print("1, 2, 3 중 하나를 입력하세요.")


def normalize_ui_choice(ui: str) -> str:
    if ui == "ask":
        return choose_ui_interactively()
    if ui == "matplotlib":
        return "mpl"
    return ui


def open_visualizer(
    ui: str,
    map_data: dict,
    records: list[dict],
    log_path: Path,
    depot_events: list[dict],
    bottleneck_events: list[dict],
    metadata: dict,
    interval_ms: int,
    speed: float,
    trace_window: int,
    map_path: Path,
):
    if ui == "none":
        print("Visualizer skipped by selection.")
        return

    if ui == "qt":
        try:
            from visualizer_qt import run_replayer
        except SystemExit as exc:
            print(exc, flush=True)
            print("Qt UI unavailable; falling back to Matplotlib.", flush=True)
        else:
            run_replayer(
                map_data,
                records,
                log_path,
                depot_events=depot_events,
                bottleneck_events=bottleneck_events,
                metadata=metadata,
                map_path=map_path,
                interval_ms=interval_ms,
                initial_speed=speed,
                trace_max_points=trace_window,
            )
            return

    from visualizer import Replayer

    replayer = Replayer(
        map_data,
        records,
        log_path,
        depot_events=depot_events,
        bottleneck_events=bottleneck_events,
        interval_ms=interval_ms,
        initial_speed=speed,
        trace_max_points=trace_window,
        map_path=map_path,
    )
    replayer.build()


def main():
    args = parse_args()

    selected_product = choose_product(args.product)
    configure_product_env(selected_product)

    from config import (
        MAP_PATH,
        PRODUCT_CODE,
        PRODUCT_LABEL,
        PRODUCT_PARTS,
        SELECTED_PRODUCT,
        SIMULATION_MAX_TIME,
    )

    max_time = float(args.max_time if args.max_time is not None else SIMULATION_MAX_TIME)
    print(f"실행 제품: {PRODUCT_LABEL} ({SELECTED_PRODUCT})", flush=True)

    map_data = _load_json(MAP_PATH)
    print(
        f"시뮬레이션 실행 중... max_time={max_time:g}초, "
        f"map={MAP_PATH.name}, motion_log_interval={args.motion_log_interval:g}초",
        flush=True,
    )
    records, depot_events, bottleneck_events = run_and_collect(
        max_time,
        map_data,
        quiet=not args.show_sim_log,
        motion_log_interval=args.motion_log_interval,
    )

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"amr_motion_{timestamp}.json"
    metadata = {
        "product": SELECTED_PRODUCT,
        "product_code": PRODUCT_CODE,
        "product_label": PRODUCT_LABEL,
        "parts": list(PRODUCT_PARTS),
        "map_file": MAP_PATH.name,
        "max_time": max_time,
        "motion_log_interval": args.motion_log_interval,
        "trace_window": args.trace_window,
    }
    save_records(
        records,
        depot_events,
        bottleneck_events,
        log_path,
        metadata=metadata,
    )

    print(f"Collected {len(records)} AMR movement records.", flush=True)
    print(f"Collected {len(depot_events)} depot occupancy events.", flush=True)
    print(f"Collected {len(bottleneck_events)} bottleneck events.", flush=True)
    print(f"Saved → {log_path}", flush=True)

    if not records:
        print("No AMR movement records were collected; visualizer skipped.")
        return

    if not args.no_show:
        ui = normalize_ui_choice(args.ui)
        open_visualizer(
            ui,
            map_data,
            records,
            log_path,
            depot_events,
            bottleneck_events,
            metadata,
            args.interval_ms,
            args.speed,
            args.trace_window,
            MAP_PATH,
        )


if __name__ == "__main__":
    main()
