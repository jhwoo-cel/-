from __future__ import annotations

import argparse
import os


PRODUCT_ENV_VAR = "PYDEVS_PRODUCT"
DEFAULT_PRODUCT = "both"

PRODUCT_SCENARIOS = {
    "both": {
        "label": "오븐+식기세척기 통합",
        "product": "ALL",
        "parts": (),
        "map_file": "oven_dishwasher_map.json",
    },
    "dishwasher": {
        "label": "식기세척기",
        "product": "A",
        "parts": ("a", "b", "c", "d", "e", "f", "g", "h"),
        "map_file": "oven_dishwasher_map.json",
    },
    "oven": {
        "label": "오븐",
        "product": "B",
        "parts": ("a", "b", "c"),
        "map_file": "oven_dishwasher_map.json",
    },
}

PRODUCT_ALIASES = {
    "": "both",
    "0": "both",
    "all": "both",
    "both": "both",
    "통합": "both",
    "전체": "both",
    "1": "dishwasher",
    "a": "dishwasher",
    "dish": "dishwasher",
    "dishwasher": "dishwasher",
    "식기세척기": "dishwasher",
    "2": "oven",
    "b": "oven",
    "oven": "oven",
    "오븐": "oven",
}


def normalize_product(value: str | None) -> str:
    if value is None:
        return DEFAULT_PRODUCT

    key = str(value).strip().lower()
    product = PRODUCT_ALIASES.get(key)
    if product is None:
        valid = ", ".join(PRODUCT_SCENARIOS)
        raise ValueError(f"Unknown product '{value}'. Choose one of: {valid}")
    return product


def product_arg(value: str) -> str:
    try:
        return normalize_product(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_product_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--product",
        "-p",
        type=product_arg,
        default=DEFAULT_PRODUCT,
        metavar="{both,dishwasher,oven}",
        help="Debug filter: both/default, dishwasher/식기세척기, or oven/오븐.",
    )


def choose_product(value: str | None) -> str:
    return normalize_product(value)


def configure_product_env(product: str) -> str:
    selected = normalize_product(product)
    os.environ[PRODUCT_ENV_VAR] = selected
    return selected
