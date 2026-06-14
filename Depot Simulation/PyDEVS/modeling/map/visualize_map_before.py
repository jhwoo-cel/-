"""
Map visualizer for depot simulation JSON map files.
Usage: python visualize_map.py [map_file.json]
"""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MAP_PATH

# map에서 amr TypeA는 대차 옮기는 amr. Type B는 박스 옮기는 amr
# part a -> Front 전방데포
# part b -> Back 후방데포

# ── visual config ──────────────────────────────────────────────────────────────
GRID_COLOR      = "#E8E8E8"
BG_COLOR        = "#FAFAFA"
REGION_ALPHA    = 0.25

NODE_STYLES = {
    # WH rack types
    "WH_O": dict(color="#2196F3", marker="s", size=120, label="WH Rack-O (occupied)"),
    "WH_X": dict(color="#90CAF9", marker="s", size=120, label="WH Rack-X (empty)"),
    # ML node types
    "ML_R": dict(color="#F44336", marker="^", size=140, label="ML Recall node"),
    "ML_P": dict(color="#FF9800", marker="D", size=120, label="ML Process node"),
    # PS (AMR Parking Station)
    "PS":   dict(color="#43A047", marker="P", size=150, label="PS AMR Parking"),
}

REGION_COLORS = [
    "#A5D6A7",  # green-ish for manufacturing line
    "#CE93D8",
    "#FFCC80",
    "#80DEEA",
    "#EF9A9A",
]
# ──────────────────────────────────────────────────────────────────────────────


def load_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_lookup(data: dict) -> dict:
    """Build a (grid_x, grid_y) → info-dict lookup for click tooltips."""
    lookup = {}

    for loc in data.get("wh_locations", []):
        gx, gy = loc["coordinates"]["x"], loc["coordinates"]["y"]
        lookup[(gx, gy)] = {
            "type": "WH",
            "id": loc["location_id"],
            "rack_type": loc["rack_type"],
            "section": loc.get("section_label", ""),
            "part": loc.get("part", ""),
            "slot": loc.get("slot", ""),
            "occupied": loc.get("occupied", 0),
        }

    for node in data.get("ml_nodes", []):
        gx, gy = node["coordinates"]["x"], node["coordinates"]["y"]
        ntype = node.get("rack_type") or node.get("node_type", "")
        lookup[(gx, gy)] = {
            "type": "ML",
            "id": node["location_id"],
            "rack_type": node.get("rack_type_label") or node.get("node_type_label") or ntype,
            "part": node.get("part") or node.get("station", ""),
            "slot": node.get("slot") or node.get("node_index", ""),
        }

    for node in data.get("ps_nodes", []):
        gx, gy = node["coordinates"]["x"], node["coordinates"]["y"]
        lookup[(gx, gy)] = {
            "type": "PS",
            "id": node["location_id"],
            "amr_type": node.get("amr_type", ""),
            "slot": node.get("slot", ""),
            "occupied": node.get("occupied", 0),
        }

    cell_map = data.get("cell_map", {})
    for key, info in cell_map.items():
        gx, gy = map(int, key.split(","))
        if (gx, gy) not in lookup:
            lookup[(gx, gy)] = {"type": "Region", "name": info.get("name", "")}

    return lookup


def _make_tooltip_text(info: dict, gx: int, gy: int) -> str:
    lines = [f"Grid  x={gx}, y={gy}"]
    t = info["type"]
    if t == "WH":
        status = "occupied" if info["occupied"] else "empty"
        lines += [
            f"Type  : Warehouse ({t})",
            f"ID    : {info['id']}",
            f"Section {info['section']} / Part {info['part']} / Slot {info['slot']}",
            f"Rack  : {info['rack_type']}  [{status}]",
        ]
    elif t == "ML":
        lines += [
            f"Type  : Manufacturing Line ({t})",
            f"ID    : {info['id']}",
            f"Part {info['part']} / {info['rack_type']} #{info['slot']}",
        ]
    elif t == "PS":
        status = "occupied" if info["occupied"] else "empty"
        lines += [
            f"Type  : AMR Parking Station ({t})",
            f"ID    : {info['id']}",
            f"AMR Type {info['amr_type']} / Slot {info['slot']}  [{status}]",
        ]
    else:
        lines += [f"Region: {info.get('name', '-')}"]
    return "\n".join(lines)


def draw(data: dict, save_path: str | None = None):
    meta   = data["metadata"]
    W, H   = meta["grid"]["width"], meta["grid"]["height"]

    lookup = _build_lookup(data)

    fig, ax = plt.subplots(figsize=(max(14, W * 0.28), max(9, H * 0.28)))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # ── grid lines ──────────────────────────────────────────────────────────
    for x in range(W + 1):
        ax.axvline(x, color=GRID_COLOR, linewidth=0.4, zorder=0)
    for y in range(H + 1):
        ax.axhline(y, color=GRID_COLOR, linewidth=0.4, zorder=0)

    # ── regions (colored fill) ──────────────────────────────────────────────
    region_patches = []
    for idx, region in enumerate(data.get("regions", [])):
        color = REGION_COLORS[idx % len(REGION_COLORS)]
        for (cx, cy) in region["cells"]:
            rect = mpatches.Rectangle(
                (cx, cy), 1, 1,
                linewidth=0, facecolor=color, alpha=REGION_ALPHA, zorder=1,
            )
            ax.add_patch(rect)
        # bounding-box label
        bb = region["bounding_box"]
        lx = bb["x_min"] + bb["width"] / 2
        ly = bb["y_min"] + bb["height"] / 2
        ax.text(
            lx, ly, region["name"],
            ha="center", va="center", fontsize=6.5, color="#444",
            style="italic", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.55, ec="none"),
        )
        region_patches.append(
            mpatches.Patch(color=color, alpha=0.6, label=f'Region: {region["name"]}')
        )

    # ── WH locations ────────────────────────────────────────────────────────
    for loc in data.get("wh_locations", []):
        cx, cy = loc["coordinates"]["x"], loc["coordinates"]["y"]
        px, py = cx + 0.5, cy + 0.5
        rtype   = loc["rack_type"]           # "O" or "X"
        key     = f"WH_{rtype}"
        style   = NODE_STYLES[key]
        # fill depends on occupied flag
        fc = style["color"] if loc["occupied"] else "#FFFFFF"
        ec = style["color"]
        ax.scatter(px, py, s=style["size"], c=fc, marker=style["marker"],
                   edgecolors=ec, linewidths=1.2, zorder=6)
        ax.text(px, py - 0.38, loc["location_id"].replace("_", "\n", 2),
                ha="center", va="top", fontsize=4.5, color="#333", zorder=7,
                linespacing=1.1)

    # ── ML nodes ────────────────────────────────────────────────────────────
    for node in data.get("ml_nodes", []):
        cx, cy = node["coordinates"]["x"], node["coordinates"]["y"]
        px, py = cx + 0.5, cy + 0.5
        # 신 스키마: rack_type/part/slot  |  구 스키마: node_type/station/node_index
        ntype = node.get("rack_type") or node.get("node_type", "")
        part  = node.get("part")      or node.get("station",   "")
        slot  = node.get("slot")      or node.get("node_index", "")
        key   = f"ML_{ntype}"
        style = NODE_STYLES.get(key, NODE_STYLES["ML_P"])
        ax.scatter(px, py, s=style["size"], c=style["color"], marker=style["marker"],
                   edgecolors="white", linewidths=0.8, zorder=6)
        label = f'{part}-{ntype}{slot}'
        ax.text(px, py + 0.38, label,
                ha="center", va="bottom", fontsize=5.5, fontweight="bold",
                color=style["color"], zorder=7)

    # ── PS nodes ────────────────────────────────────────────────────────────
    style_ps = NODE_STYLES["PS"]
    for node in data.get("ps_nodes", []):
        cx, cy = node["coordinates"]["x"], node["coordinates"]["y"]
        px, py = cx + 0.5, cy + 0.5
        amr  = node.get("amr_type", "")
        slot = node.get("slot", "")
        fc = style_ps["color"] if node.get("occupied") else "#FFFFFF"
        ax.scatter(px, py, s=style_ps["size"], c=fc, marker=style_ps["marker"],
                   edgecolors=style_ps["color"], linewidths=1.2, zorder=6)
        label = f'{amr}-{slot}'
        ax.text(px, py + 0.38, label,
                ha="center", va="bottom", fontsize=5.5, fontweight="bold",
                color=style_ps["color"], zorder=7)

    # ── axes / labels ────────────────────────────────────────────────────────
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")

    # x-axis ticks every 5 cells
    xticks = range(0, W + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=7)

    # y-axis: Y=0 at bottom, increases upward
    ytick_vals = range(0, H + 1, 5)
    ax.set_yticks(list(ytick_vals))
    ax.set_yticklabels(list(ytick_vals), fontsize=7)

    ax.set_xlabel("X (grid column)", fontsize=8)
    ax.set_ylabel("Y (grid row, 0 = bottom)", fontsize=8)

    # ── legend ──────────────────────────────────────────────────────────────
    legend_handles = region_patches.copy()
    for key, s in NODE_STYLES.items():
        legend_handles.append(
            plt.Line2D([0], [0], marker=s["marker"], color="w",
                       markerfacecolor=s["color"], markeredgecolor=s["color"],
                       markersize=7, label=s["label"])
        )
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=6.5, framealpha=0.85, edgecolor="#CCC")

    # title
    created = data["metadata"].get("created", "")[:19]
    ax.set_title(
        f'Depot Map  ({W}×{H} grid)   [{created}]',
        fontsize=11, fontweight="bold", pad=10,
    )

    # ── click tooltip ────────────────────────────────────────────────────────
    tooltip_box = ax.text(
        0, 0, "", transform=ax.transData,
        fontsize=7.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#FBC02D",
                  alpha=0.92, linewidth=1.2),
        zorder=20, visible=False,
    )
    highlight = mpatches.Rectangle((0, 0), 1, 1,
        linewidth=1.5, edgecolor="#FBC02D", facecolor="#FFF9C4",
        alpha=0.5, zorder=15, visible=False,
    )
    ax.add_patch(highlight)

    def on_click(event):
        if event.inaxes is not ax:
            return
        gx = int(event.xdata)
        gy = int(event.ydata)          # plot-y == map-y (same direction)

        # keep highlight visible even for empty cells
        highlight.set_xy((gx, gy))
        highlight.set_visible(True)

        info = lookup.get((gx, gy))
        text = _make_tooltip_text(info, gx, gy) if info else f"Grid  x={gx}, y={gy}\n(empty)"

        # X: 우측 끝 근처면 왼쪽으로
        tx = gx + 1.1 if gx < W - 12 else gx - 0.1
        ha = "left" if gx < W - 12 else "right"
        # Y: 상단(y 큰 값) 근처면 아래로, 그 외엔 위로
        if gy >= H - 6:
            ty, va = gy - 0.1, "top"
        else:
            ty, va = gy + 1.1, "bottom"
        tooltip_box.set_position((tx, ty))
        tooltip_box.set_ha(ha)
        tooltip_box.set_va(va)
        tooltip_box.set_text(text)
        tooltip_box.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()


def main():
    default_map = str(MAP_PATH)
    path = sys.argv[1] if len(sys.argv) > 1 else default_map

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    data = load_map(path)
    save_png = path.replace(".json", ".png")
    draw(data, save_path=save_png)


if __name__ == "__main__":
    main()
