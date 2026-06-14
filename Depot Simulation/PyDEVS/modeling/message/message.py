# -*- coding: utf-8 -*-
"""Unified message class for depot DEVS events.

`Message` is the single message type used across all atomic models.
Construct with `Message(field=value, ...)` and modify via
`message.update(...)`, which returns a deep-copied instance — the
original is never mutated.

Note: the `from` field is exposed as `from_` in Python (since `from`
is a reserved keyword) but is serialized as `from` in dict/JSON
output.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any
import json


_FROM_PY = "from_"
_FROM_KEY = "from"


@dataclass
class Message:

    # supply order generator, part update generator, mcs에서 사용
    product : Any = None
    part : Any = None
    cart_count: Any = None
    generated_time: Any = None
    info_type : Any = None
    from_: Any = None
    to: Any = None # 여기까지


    AMR_id: Any = None
    task_id: Any = None
    task_type: Any = None
    task_step: Any = None
    current_location: Any = None
    goal_type: Any = None
    goal_location: Any = None
    route: Any = None
    start_time: Any = None
    end_time: Any = None
    timestamp: Any = None
    travel_time: Any = None
    travel_distance: Any = None
    speed: Any = None
    delay_time: Any = None
    battery_level: Any = None
    amr_status: Any = None
    result_status: Any = None
    event_description: Any = None
    current_situation: Any = None
    arrival_event: Any = None
    idle_type: Any = None
    next_destination: Any = None
    next_action: Any = None
    move_num: Any = None
    cart: Any = None
    arrive_interval: Any = None
    load_type: Any = None
    carrying_type: Any = None
    location_id: Any = None
    location_code: Any = None
    node: Any = None
    x: Any = None
    y: Any = None

    access_node_id: Any = None
    type: Any = None
    capacity: Any = None
    status: Any = None

    target_slot: Any = None
    target_label: Any = None
    chain_recovery: Any = None
    recovery_from: Any = None
    recovery_to: Any = None
    recovery_ready_time: Any = None
    recovery_wait_time: Any = None
    initial_leg: Any = None
    amr_product: Any = None
    process_status: Any = None
    process_completed_time: Any = None

    bottleneck_event: Any = None
    bottleneck_event_type: Any = None
    bottleneck_depot_type: Any = None
    bottleneck_current: Any = None
    bottleneck_total: Any = None
    bottleneck_delta_current: Any = None
    bottleneck_delta_total: Any = None

    amr_list: Any = None
    item_list: Any = None
    amr_type: Any = None
    current_coord: Any = None
    from_coord: Any = None
    to_coord: Any = None
    station_coord: Any = None

    @classmethod
    def field_names(cls):
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_dict(cls, data, strict=False):
        data = dict(data) if data else {}
        if _FROM_KEY in data and _FROM_PY not in data:
            data[_FROM_PY] = data.pop(_FROM_KEY)

        names = set(cls.field_names())
        if strict:
            unknown = set(data) - names
            if unknown:
                raise KeyError(
                    f"Unknown message keys: {', '.join(sorted(unknown))}"
                )

        kwargs = {k: deepcopy(v) for k, v in data.items() if k in names}
        return cls(**kwargs)

    def update(self, values=None, strict=False, **kwargs):
        merged = {}
        if values is not None:
            merged.update(values)
        merged.update(kwargs)
        if _FROM_KEY in merged and _FROM_PY not in merged:
            merged[_FROM_PY] = merged.pop(_FROM_KEY)

        names = set(self.field_names())
        if strict:
            unknown = set(merged) - names
            if unknown:
                raise KeyError(
                    f"Unknown message keys: {', '.join(sorted(unknown))}"
                )

        new_msg = self.copy()
        for key, value in merged.items():
            if key in names:
                setattr(new_msg, key, deepcopy(value))
        return new_msg

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        data = asdict(self)
        ordered = {}
        for name in self.field_names():
            key = _FROM_KEY if name == _FROM_PY else name
            ordered[key] = data[name]
        return ordered

    def to_json(self, indent=2, ensure_ascii=False):
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=ensure_ascii
        )

    def save_json(self, path, indent=2, ensure_ascii=False):
        output_path = Path(path)
        output_path.write_text(
            self.to_json(indent=indent, ensure_ascii=ensure_ascii),
            encoding="utf-8",
        )
        return output_path
