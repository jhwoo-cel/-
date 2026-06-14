"""
Map visualizer for depot simulation JSON map files.
Usage: python visualize_map.py [map_file.json]
"""

import json
import sys
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection


logging.getLogger("matplotlib").setLevel(logging.ERROR)


GRID_COLOR      = "#E8E8E8"
BG_COLOR        = "#FAFAFA"
REGION_ALPHA    = 0.25
PATH_NODE_SIZE  = 2          
CORE_NODE_SIZE  = 32


NODE_STYLES = {
    # WH rack types
    "WH_O": dict(color="#2196F3", marker="s", size=120, label="WH Rack-O (occupied)"),
    "WH_X": dict(color="#90CAF9", marker="s", size=120, label="WH Rack-X (empty)"),
    # ML node types
    "ML_R": dict(color="#F44336", marker="^", size=140, label="ML Recall node"),
    "ML_P": dict(color="#FF9800", marker="D", size=120, label="ML Process node"),
    # PS (AMR Parking Station)
    "PS":   dict(color="#43A047", marker="P", size=150, label="PS AMR Parking"),
    # AMR path / route nodes
    "PATH": dict(color="#E53935", marker="o", size=70, label="AMR Path node"),
    "CORE_PATH": dict(color="#111827", marker="*", size=95, label="Core AMR Path node"),
}

REGION_COLORS = [
    "#A5D6A7",  # green-ish for manufacturing line
    "#CE93D8",
    "#FFCC80",
    "#80DEEA",
    "#EF9A9A",
]

PRODUCT_LABELS = {
    "A": "Dishwasher",
    "B": "Oven",
}

PART_LABELS_BY_PRODUCT = {
    "A": {
        "a": "Control Panel",
        "b": "Door Liner",
        "c": "Front Cover",
        "d": "Cabinet",
        "e": "Damper",
        "f": "Filter",
        "g": "Mesh",
        "h": "Lower Cover",
    },
    "B": {
        "a": "Cooktop",
        "b": "Drawer",
        "c": "Door",
    },
}


def load_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _display_product(product: str) -> str:
    label = PRODUCT_LABELS.get(product)
    return f"{product} ({label})" if label else product


def _display_part(product: str, part: str) -> str:
    label = PART_LABELS_BY_PRODUCT.get(product, {}).get(part)
    return f"{part} ({label})" if label else part


def load_core_amr_nodes(map_path: str) -> list[dict]:
    core_path = os.path.join(os.path.dirname(map_path), "core_amr_node.json")
    if not os.path.exists(core_path):
        print(f"[WARN] Core AMR node file not found: {core_path}")
        return []

    core_data = load_map(core_path)
    nodes = core_data.get("core_amr_path_nodes", [])
    print(f"Loaded {len(nodes)} core AMR path nodes from {core_path}")
    return nodes


def _build_lookup(data: dict) -> dict:
    """Build a (grid_x, grid_y) ??info-dict lookup for click tooltips."""
    lookup = {}

    for loc in data.get("wh_locations", []):
        gx, gy = loc["coordinates"]["x"], loc["coordinates"]["y"]
        product = loc.get("product", "")
        part = loc.get("part", "")
        lookup[(gx, gy)] = {
            "type": "WH",
            "id": loc["location_id"],
            "product": product,
            "rack_type": loc["rack_type"],
            "section": loc.get("section_label", ""),
            "part": _display_part(product, part),
            "slot": loc.get("slot", ""),
            "occupied": loc.get("occupied", 0),
        }

    for node in data.get("ml_nodes", []):
        gx, gy = node["coordinates"]["x"], node["coordinates"]["y"]
        ntype = node.get("rack_type") or node.get("node_type", "")
        product = node.get("product", "")
        part = node.get("part") or node.get("station", "")
        lookup[(gx, gy)] = {
            "type": "ML",
            "id": node["location_id"],
            "product": product,
            "rack_type": node.get("rack_type_label") or node.get("node_type_label") or ntype,
            "part": _display_part(product, part),
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

    for node in data.get("amr_path_nodes", []):
        gx, gy = node["x"], node["y"]
        lookup[(gx, gy)] = {
            "type": "PATH",
            "id": node.get("name", ""),
        }

    for node in data.get("core_amr_path_nodes", []):
        gx, gy = node["x"], node["y"]
        lookup[(gx, gy)] = {
            "type": "CORE_PATH",
            "id": node.get("name", ""),
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
            f"Product: {_display_product(info.get('product', ''))}",
            f"Section {info['section']} / Part {info['part']} / Slot {info['slot']}",
            f"Rack  : {info['rack_type']}  [{status}]",
        ]
    elif t == "ML":
        lines += [
            f"Type  : Manufacturing Line ({t})",
            f"ID    : {info['id']}",
            f"Product: {_display_product(info.get('product', ''))}",
            f"Part {info['part']} / {info['rack_type']} #{info['slot']}",
        ]
    elif t == "PS":
        status = "occupied" if info["occupied"] else "empty"
        lines += [
            f"Type  : AMR Parking Station ({t})",
            f"ID    : {info['id']}",
            f"AMR Type {info['amr_type']} / Slot {info['slot']}  [{status}]",
        ]
    elif t == "PATH":
        lines += [
            f"Type  : AMR Path node ({t})",
            f"ID    : {info['id']}",
        ]
    elif t == "CORE_PATH":
        lines += [
            "Type  : Core AMR Path node",
            f"ID    : {info['id']}",
        ]
    else:
        lines += [f"Region: {info.get('name', '-')}"]
    return "\n".join(lines)


def draw(data: dict, save_path: str | None = None):
    meta   = data["metadata"]
    W, H   = meta["grid"]["width"], meta["grid"]["height"]

    lookup = _build_lookup(data)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    grid_segs = [[(x, 0), (x, H)] for x in range(W + 1)]
    grid_segs += [[(0, y), (W, y)] for y in range(H + 1)]
    ax.add_collection(LineCollection(grid_segs, colors=GRID_COLOR,
                                     linewidths=0.4, zorder=0))

    region_patches = []
    region_rects = []
    region_facecolors = []
    for idx, region in enumerate(data.get("regions", [])):
        color = REGION_COLORS[idx % len(REGION_COLORS)]
        for (cx, cy) in region["cells"]:
            region_rects.append(mpatches.Rectangle((cx, cy), 1, 1))
            region_facecolors.append(color)
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
    if region_rects:
        ax.add_collection(PatchCollection(
            region_rects, facecolors=region_facecolors,
            alpha=REGION_ALPHA, linewidths=0, zorder=1,
        ))

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

    for node in data.get("ml_nodes", []):
        cx, cy = node["coordinates"]["x"], node["coordinates"]["y"]
        px, py = cx + 0.5, cy + 0.5
        # ?????熬곥굥??묅뵾? rack_type/part/slot  |  ?????熬곥굥??묅뵾? node_type/station/node_index
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

    style_path = NODE_STYLES["PATH"]

    def _path_index(n):
        s = str(n.get("name", "")).lstrip("Nn")
        return int(s) if s.isdigit() else 1 << 30

    PATH_LABEL_LIMIT = 300
    path_nodes = sorted(data.get("amr_path_nodes", []), key=_path_index)
    if path_nodes:
        pxs = [n["x"] + 0.5 for n in path_nodes]
        pys = [n["y"] + 0.5 for n in path_nodes]
        ax.scatter(pxs, pys, s=PATH_NODE_SIZE, c=style_path["color"],
                   marker=style_path["marker"], edgecolors="none",
                   linewidths=0, zorder=6)
        if len(path_nodes) <= PATH_LABEL_LIMIT:
            for n, px, py in zip(path_nodes, pxs, pys):
                ax.text(px, py + 0.36, n.get("name", ""),
                        ha="center", va="bottom", fontsize=5,
                        fontweight="bold", color=style_path["color"], zorder=7)

    core_path_nodes = sorted(data.get("core_amr_path_nodes", []), key=_path_index)
    if core_path_nodes:
        style_core = NODE_STYLES["CORE_PATH"]
        cxs = [n["x"] + 0.5 for n in core_path_nodes]
        cys = [n["y"] + 0.5 for n in core_path_nodes]
        ax.scatter(cxs, cys, s=CORE_NODE_SIZE, c=style_core["color"],
                   marker=style_core["marker"], edgecolors="#FFFFFF",
                   linewidths=0.45, zorder=8)

        CORE_LABEL_LIMIT = 800
        if len(core_path_nodes) <= CORE_LABEL_LIMIT:
            for n, px, py in zip(core_path_nodes, cxs, cys):
                ax.text(px, py + 0.46, n.get("name", ""),
                        ha="center", va="bottom", fontsize=4.5,
                        fontweight="bold", color=style_core["color"], zorder=9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)

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
        f'Depot Map  ({W}x{H} grid)   [{created}]',
        fontsize=11, fontweight="bold", pad=10,
    )

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
        if event.inaxes is not ax or event.button != 1:
            return
        gx = int(event.xdata)
        gy = int(event.ydata)

        highlight.set_xy((gx, gy))
        highlight.set_visible(True)

        info = lookup.get((gx, gy))
        text = _make_tooltip_text(info, gx, gy) if info else f"Grid  x={gx}, y={gy}\n(empty)"

        tx = gx + 1.1 if gx < W - 12 else gx - 0.1
        ha = "left" if gx < W - 12 else "right"
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

    def on_scroll(event):
        if event.inaxes is not ax:
            return
        base = 1.25
        factor = (1 / base) if event.button == "up" else base   # up = ?癲?
        xdata, ydata = event.xdata, event.ydata
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        new_w = (x1 - x0) * factor
        new_h = (y1 - y0) * factor
        relx = (xdata - x0) / (x1 - x0)
        rely = (ydata - y0) / (y1 - y0)
        ax.set_xlim(xdata - new_w * relx, xdata + new_w * (1 - relx))
        ax.set_ylim(ydata - new_h * rely, ydata + new_h * (1 - rely))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    pan = {"active": False, "x": 0, "y": 0, "xlim": None, "ylim": None}

    def on_pan_press(event):
        if event.button == 3:
            pan.update(active=True, x=event.x, y=event.y,
                       xlim=ax.get_xlim(), ylim=ax.get_ylim())

    def on_pan_move(event):
        if not pan["active"] or event.x is None:
            return
        bbox = ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return
        xl0, xl1 = pan["xlim"]
        yl0, yl1 = pan["ylim"]
        dx = (event.x - pan["x"]) / bbox.width * (xl1 - xl0)
        dy = (event.y - pan["y"]) / bbox.height * (yl1 - yl0)
        ax.set_xlim(xl0 - dx, xl1 - dx)
        ax.set_ylim(yl0 - dy, yl1 - dy)
        fig.canvas.draw_idle()

    def on_pan_release(event):
        if event.button == 3:
            pan["active"] = False

    fig.canvas.mpl_connect("button_press_event",   on_pan_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_pan_move)
    fig.canvas.mpl_connect("button_release_event", on_pan_release)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ??{save_path}")

    plt.show()


def main():
    default_map = os.path.join(os.path.dirname(__file__), "oven_dishwasher_map.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default_map

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    data = load_map(path)
    data["core_amr_path_nodes"] = load_core_amr_nodes(path)
    save_png = path.replace(".json", ".png")
    draw(data, save_path=save_png)


if __name__ == "__main__":
    main()
