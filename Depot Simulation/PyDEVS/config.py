import os
from pathlib import Path

from runtime_product import DEFAULT_PRODUCT, PRODUCT_ENV_VAR, normalize_product


# ---------------------------------------------------------------------------
# Product / map selection
# ---------------------------------------------------------------------------
SELECTED_PRODUCT = normalize_product(os.environ.get(PRODUCT_ENV_VAR, DEFAULT_PRODUCT))

PRODUCTS = {
    "dishwasher": {
        "label": "식기세척기",
        "code": "A",
        "parts": ("a", "b", "c", "d", "e", "f", "g", "h"),
    },
    "oven": {
        "label": "오븐",
        "code": "B",
        "parts": ("a", "b", "c"),
    },
}

ACTIVE_PRODUCT_KEYS = (
    ("dishwasher", "oven")
    if SELECTED_PRODUCT == "both"
    else (SELECTED_PRODUCT,)
)
ACTIVE_PRODUCT_CODES = tuple(PRODUCTS[key]["code"] for key in ACTIVE_PRODUCT_KEYS)

PRODUCT_LABEL = (
    "오븐+식기세척기 통합"
    if SELECTED_PRODUCT == "both"
    else PRODUCTS[SELECTED_PRODUCT]["label"]
)
PRODUCT_CODE = "ALL" if SELECTED_PRODUCT == "both" else PRODUCTS[SELECTED_PRODUCT]["code"]
PRODUCT_PARTS = tuple(
    (PRODUCTS[key]["code"], part)
    for key in ACTIVE_PRODUCT_KEYS
    for part in PRODUCTS[key]["parts"]
)

MAP_FILE = "oven_dishwasher_map.json"
MAP_PATH = Path(__file__).resolve().parent / "modeling" / "map" / MAP_FILE

PRODUCT_PART_LABELS_BY_PRODUCT = {
    "dishwasher": {
        "a": "Control Panel",
        "b": "Door Liner",
        "c": "Front Cover",
        "d": "Cabinet",
        "e": "Damper",
        "f": "Filter",
        "g": "Mesh",
        "h": "Lower Cover",
    },
    "oven": {
        "a": "Cooktop",
        "b": "Drawer",
        "c": "Door",
    },
}
PRODUCT_PART_LABELS = {
    part: label
    for key in ACTIVE_PRODUCT_KEYS
    for part, label in PRODUCT_PART_LABELS_BY_PRODUCT[key].items()
}


# ---------------------------------------------------------------------------
# Time scale
# Simulation time is modeled in real seconds.
# 8 real hours = 28,800 simulation seconds.
# ---------------------------------------------------------------------------
SIMULATION_MAX_TIME = 8 * 60 * 60
SIM_SEC_PER_REAL_HOUR = 60 * 60


# ---------------------------------------------------------------------------
# AMR movement
# ---------------------------------------------------------------------------
AMR_LOAD_UNLOAD_TIME = 1.25 * 60
MAP_CELL_SIZE_M = 0.2

AMR_MAX_SPEED_M_PER_SEC = 1.1
AMR_LOADED_MAX_SPEED_M_PER_SEC = AMR_MAX_SPEED_M_PER_SEC
AMR_UNLOADED_MAX_SPEED_M_PER_SEC = AMR_MAX_SPEED_M_PER_SEC
AMR_EMPTY_CART_SPEED_FACTOR = 1.0
AMR_EMPTY_CART_MAX_SPEED_M_PER_SEC = (
    AMR_LOADED_MAX_SPEED_M_PER_SEC * AMR_EMPTY_CART_SPEED_FACTOR
)
AMR_ACCELERATION_M_PER_SEC2 = 0.5
AMR_DECELERATION_M_PER_SEC2 = 0.5
# Docking speed used only near load/unload target depots.
AMR_DOCKING_SPEED_M_PER_SEC = 0.2
# Maximum docking approach/departure distance. If the straight approach segment
# is shorter than this, AMR uses only the available segment distance.
AMR_DOCKING_DISTANCE_M = 1.0
AMR_TURN_TIME_SEC = 0.0
AMR_ROUTE_POINT_MODE = "core_turn"
AMR_PHYSICAL_LOG_INTERVAL_SEC = 1.0

# ACS reservation is integer tick based; actual AMR motion uses the physical
# speed profile in AMR.
AMR_CELL_TRAVEL_TIME = max(1.0, MAP_CELL_SIZE_M / AMR_MAX_SPEED_M_PER_SEC)
AMR_SPEED_M_PER_SEC = AMR_MAX_SPEED_M_PER_SEC
AMR_SPEED_M_PER_MIN = AMR_SPEED_M_PER_SEC * 60

AMR_START_LOCATIONS_BY_PRODUCT_CODE = {
    "A": (
        "A_PS_TypeA_1",
        "A_PS_TypeA_2",
        "A_PS_TypeA_3",
        "A_PS_TypeA_4",
    ),
    "B": (
        "B_PS_TypeA_1",
        "B_PS_TypeA_2",
        "B_PS_TypeA_3",
        "B_PS_TypeA_4",
        "B_PS_TypeA_5",
        "B_PS_TypeA_6",
        "B_PS_TypeA_7",
        "B_PS_TypeA_8",
    ),
}

AMR_LIST = [
    {
        "AMR_id": f"{product_code}_AMR_{index:02d}",
        "product": product_code,
        "speed": AMR_SPEED_M_PER_SEC,
        "loaded_max_speed": AMR_LOADED_MAX_SPEED_M_PER_SEC,
        "unloaded_max_speed": AMR_UNLOADED_MAX_SPEED_M_PER_SEC,
        "empty_cart_max_speed": AMR_EMPTY_CART_MAX_SPEED_M_PER_SEC,
        "docking_speed": AMR_DOCKING_SPEED_M_PER_SEC,
        "docking_distance": AMR_DOCKING_DISTANCE_M,
        "current_location": location,
        "battery_level": 100.0,
        "physical_log_interval": AMR_PHYSICAL_LOG_INTERVAL_SEC,
    }
    for product_code in ACTIVE_PRODUCT_CODES
    for index, location in enumerate(
        AMR_START_LOCATIONS_BY_PRODUCT_CODE[product_code],
        start=1,
    )
]


# ---------------------------------------------------------------------------
# Supply order frequency per real hour.
# ---------------------------------------------------------------------------
PART_FREQUENCY_PER_HOUR_BY_PRODUCT_CODE = {
    "A": {
        # a: Control Panel
        "a": 2.6,
        # b: Door Liner
        "b": 12,
        # c: Front Cover
        "c": 7.8,
        # d: Cabinet
        "d": 14.4,
        # e: Damper
        "e": 0.8,
        # f: Filter
        "f": 0.1,
        # g: Mesh
        "g": 0.4,
        # h: Lower Cover
        "h": 1.1,
    },
    "B": {
        # a: Cooktop
        "a": 8,
        # b: Drawer
        "b": 4.7,
        # c: Door_H / Door_L
        "c": 4.2,
    },
}

PART_FREQUENCY_PER_HOUR = {
    (product_code, part): PART_FREQUENCY_PER_HOUR_BY_PRODUCT_CODE[product_code][part]
    for product_code, part in PRODUCT_PARTS
}

PART_INTERVAL = {
    key: SIM_SEC_PER_REAL_HOUR / freq
    for key, freq in PART_FREQUENCY_PER_HOUR.items()
}

PART_WORK_TIME = dict(PART_INTERVAL)


def _target_specs(product_code, part):
    if product_code == "B" and part == "c":
        return (
            {"target_slot": 1, "target_label": "Door_H", "chain_recovery": True},
            {"target_slot": 2, "target_label": "Door_L", "chain_recovery": True},
        )
    return (
        {"target_slot": None, "target_label": None, "chain_recovery": True},
    )


SUPPLY_ORDER_GROUPS = [
    {
        "product": product_code,
        "part": part,
        "cart_count": 1,
        "interval": PART_INTERVAL[(product_code, part)],
        "targets": _target_specs(product_code, part),
    }
    for product_code, part in PRODUCT_PARTS
]

# Compatibility table for older code paths. New code should use
# SUPPLY_ORDER_GROUPS because A/B part names overlap.
SUPPLY_ORDERS = {
    (product_code, part): {"product": product_code, "cart_count": 1}
    for product_code, part in PRODUCT_PARTS
}


# Item replenishment is intentionally disabled for the integrated baseline.
ENABLE_ITEM_REPLENISHMENT = False
ITEM_UPDATES = []
