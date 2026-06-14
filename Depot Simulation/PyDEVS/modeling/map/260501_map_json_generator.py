#!/usr/bin/env python3
"""
Warehouse Map JSON Generator
노드 ID 체계:
  WH 위치  → {product}_{zone}_{section}_{part}_{rack_type}_{slot}   예: A_WH_B_d_O_2
  ML 노드  → {product}_{zone}_{part}_{rack_type}{slot}               예: A_ML_a_R1
  PS 노드  → {product}_{zone}_{amr_type}_{slot}                      예: A_PS_TypeA_1
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import json
from datetime import datetime

# ── 상수 ───────────────────────────────────────────────────────────────────────
CELL_SIZE  = 24
MIN_CELL   = 8
MAX_CELL   = 64
DEFAULT_W  = 60
DEFAULT_H  = 40
GRID_COLOR = "#CCCCCC"
SEL_COLOR  = "#0078D4"
BG_COLOR   = "#FFFFFF"

# Zone별 기본 색상
ZONE_COLORS = {
    "WH_F":  "#A8D8EA",   # 창고 Front
    "WH_B":  "#FFCC99",   # 창고 Back
    "ML":    "#D7B8F3",   # 제조라인
    "PS":    "#C8F0C8",   # AMR Parking Station
    "other": "#B5EAD7",   # 기타
}

# AMR 타입: 코드 → (레이블, 설명)
AMR_TYPES: dict[str, tuple[str, str]] = {
    "TypeA": ("Type-A", "Type-A AMR"),
    "TypeB": ("Type-B", "Type-B AMR"),
    "TypeC": ("Type-C", "Type-C AMR"),
}

# 랙 타입: 코드 → (레이블, 설명)
RACK_TYPES: dict[str, tuple[str, str]] = {
    "O": ("O",  "실대차 Depot"),
    "X": ("X",   "공대차 Depot"),
}

# ML 노드 타입: 코드 → (레이블, 설명)
ML_NODE_TYPES: dict[str, tuple[str, str]] = {
    "R": ("Recall",     "Recall Pos   — 회수 데포"),
    "P": ("Process",   "Process Pos.    — 공정 데포"),

}

# 구역(Region) 프리셋
REGION_PRESETS: dict[str, tuple[str, str]] = {
    "aisle":      ("#D8D8D8", "통로 (Aisle)"),
    "wh_area":    ("#C5E8F7", "창고 구역 (WH Area)"),
    "ml_area":    ("#E6D5F5", "제조라인 구역 (ML Area)"),
    "obstacle":   ("#A0522D", "장애물 (Obstacle)"),
    "restricted": ("#FF8080", "제한구역 (Restricted)"),
    "charging":   ("#FFD700", "충전소 (Charging)"),
    "custom":     ("#F0F0F0", "사용자정의 (Custom)"),
}


# ── 메인 에디터 ────────────────────────────────────────────────────────────────
class MapEditor:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Warehouse Map JSON Generator")
        self.root.geometry("1400x840")
        self.root.minsize(960, 640)

        self.grid_w    = DEFAULT_W
        self.grid_h    = DEFAULT_H
        self.cell_size = CELL_SIZE
        self.regions   : dict[str, dict] = {}
        self.cell_map  : dict[tuple, str] = {}
        self.nodes     : dict[str, dict] = {}
        self.next_id   = 1

        self.sel_start = None
        self.sel_rect  = None
        self.sel_cells : set[tuple] = set()

        self._build_ui()
        self._redraw()

    # ── UI 구성 ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_menu()
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)
        self._build_toolbar(outer)
        self._build_canvas(outer)
        self._build_sidebar(outer)

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 bd=1, relief=tk.SUNKEN, anchor=tk.W,
                 font=("Arial", 9)).pack(side=tk.BOTTOM, fill=tk.X)
        self._update_status()

    def _build_menu(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="새로 만들기  Ctrl+N", command=self._new_map)
        fm.add_command(label="JSON 열기  Ctrl+O",    command=self._open_json)
        fm.add_command(label="JSON 내보내기  Ctrl+E", command=self._export_json)
        fm.add_separator()
        fm.add_command(label="종료", command=self.root.quit)
        mb.add_cascade(label="파일", menu=fm)
        sm = tk.Menu(mb, tearoff=0)
        sm.add_command(label="그리드 크기 설정", command=self._dlg_grid_size)
        mb.add_cascade(label="설정", menu=sm)
        self.root.config(menu=mb)
        self.root.bind("<Control-n>", lambda _: self._new_map())
        self.root.bind("<Control-o>", lambda _: self._open_json())
        self.root.bind("<Control-e>", lambda _: self._export_json())

    def _build_toolbar(self, parent):
        tb = tk.Frame(parent, width=82, bg="#EBEBEB", relief=tk.RIDGE, bd=1)
        tb.pack(side=tk.LEFT, fill=tk.Y)
        tb.pack_propagate(False)

        tk.Label(tb, text="도구", bg="#EBEBEB",
                 font=("Arial", 8, "bold")).pack(pady=(10, 2))
        self.tool_var = tk.StringVar(value="region")
        for icon, lbl, val in [("▭", "구역", "region"),
                                ("●", "노드", "node"),
                                ("✕", "지우개", "erase")]:
            tk.Radiobutton(
                tb, text=f"{icon}\n{lbl}", variable=self.tool_var, value=val,
                indicatoron=False, width=5, height=2,
                bg="#EBEBEB", selectcolor=SEL_COLOR, fg="black",
                font=("Arial", 8), command=self._update_status,
            ).pack(pady=3, padx=6)

        ttk.Separator(tb, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6, padx=6)
        tk.Label(tb, text="확대", bg="#EBEBEB", font=("Arial", 8)).pack()
        tk.Button(tb, text="+", width=3, command=self._zoom_in).pack(pady=1)
        tk.Button(tb, text="−", width=3, command=self._zoom_out).pack(pady=1)
        ttk.Separator(tb, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6, padx=6)
        tk.Button(
            tb, text="JSON\n내보내기", width=7,
            bg=SEL_COLOR, fg="white", font=("Arial", 8, "bold"),
            command=self._export_json,
        ).pack(pady=4, padx=6)

    def _build_canvas(self, parent):
        cf = tk.Frame(parent, relief=tk.SUNKEN, bd=1)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cf, bg=BG_COLOR, cursor="crosshair")
        vsc = ttk.Scrollbar(cf, orient=tk.VERTICAL,   command=self.canvas.yview)
        hsc = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsc.set, xscrollcommand=hsc.set)
        hsc.pack(side=tk.BOTTOM, fill=tk.X)
        vsc.pack(side=tk.RIGHT,  fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>",      self._on_wheel)
        self.canvas.bind("<Button-4>",        self._on_wheel)
        self.canvas.bind("<Button-5>",        self._on_wheel)

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, width=275, bg="#F6F6F6", relief=tk.RIDGE, bd=1)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sb.pack_propagate(False)

        tk.Label(sb, text="레이어 / 노드 목록", bg="#F6F6F6",
                 font=("Arial", 10, "bold")).pack(pady=(8, 2))
        ttk.Separator(sb, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6)

        tf = tk.Frame(sb, bg="#F6F6F6")
        tf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.tree = ttk.Treeview(
            tf, columns=("id", "zone", "pos"),
            show="headings", height=22,
        )
        self.tree.heading("id",   text="Location ID")
        self.tree.heading("zone", text="Zone")
        self.tree.heading("pos",  text="좌표")
        self.tree.column("id",   width=140, anchor=tk.W)
        self.tree.column("zone", width=55,  anchor=tk.CENTER)
        self.tree.column("pos",  width=60,  anchor=tk.CENTER)

        tsc = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsc.set)
        tsc.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<Double-1>", lambda _: self._edit_selected())
        self.tree.bind("<Delete>",   lambda _: self._delete_selected())

        bf = tk.Frame(sb, bg="#F6F6F6")
        bf.pack(fill=tk.X, padx=6, pady=4)
        tk.Button(bf, text="편집", width=7, command=self._edit_selected).pack(side=tk.LEFT, padx=2)
        tk.Button(bf, text="삭제", width=7, fg="red",
                  command=self._delete_selected).pack(side=tk.LEFT, padx=2)

        # 범례
        lf = tk.LabelFrame(sb, text="Zone 범례", bg="#F6F6F6", font=("Arial", 8))
        lf.pack(fill=tk.X, padx=6, pady=(2, 4))
        for lbl, color in [("WH Front (F)", ZONE_COLORS["WH_F"]),
                            ("WH Back  (B)", ZONE_COLORS["WH_B"]),
                            ("ML",           ZONE_COLORS["ML"]),
                            ("PS  주차장",    ZONE_COLORS["PS"]),
                            ("기타 구역",      ZONE_COLORS["other"])]:
            row = tk.Frame(lf, bg="#F6F6F6"); row.pack(anchor=tk.W, padx=6, pady=1)
            tk.Label(row, bg=color, width=2, height=1, relief=tk.RAISED).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(row, text=lbl, bg="#F6F6F6", font=("Arial", 8)).pack(side=tk.LEFT)

        tk.Label(sb, text="드래그→구역  클릭→노드\n더블클릭→편집  Del→삭제",
                 bg="#F6F6F6", font=("Arial", 7), fg="#888").pack(pady=4)

    # ── 그리기 ─────────────────────────────────────────────────────────────────

    def _redraw(self):
        c, cs = self.canvas, self.cell_size
        cw, ch = self.grid_w * cs, self.grid_h * cs
        c.delete("all")
        c.configure(scrollregion=(0, 0, cw, ch))

        for (gx, gy), rid in self.cell_map.items():
            if rid in self.regions:
                x0 = gx * cs
                y0 = (self.grid_h - 1 - gy) * cs
                c.create_rectangle(x0, y0, x0+cs, y0+cs,
                                   fill=self.regions[rid]["color"],
                                   outline="", tags="cell")

        for nd in self.nodes.values():
            self._draw_node_shape(nd)

        for i in range(self.grid_w + 1):
            c.create_line(i*cs, 0, i*cs, ch, fill=GRID_COLOR, tags="grid")
        for j in range(self.grid_h + 1):
            c.create_line(0, j*cs, cw, j*cs, fill=GRID_COLOR, tags="grid")

        if cs >= 14:
            done_r: set[str] = set()
            for (gx, gy), rid in self.cell_map.items():
                if rid in done_r or rid not in self.regions:
                    continue
                done_r.add(rid)
                cells = self.regions[rid]["cells"]
                cx = sum(p[0] for p in cells) / len(cells)
                cy = sum(p[1] for p in cells) / len(cells)
                canvas_cy = self.grid_h - 1 - cy
                c.create_text(
                    (cx+0.5)*cs, (canvas_cy+0.5)*cs,
                    text=self.regions[rid]["name"],
                    font=("Arial", max(6, cs//4)),
                    fill="#111", tags="label",
                )
            for nd in self.nodes.values():
                canvas_ndy = self.grid_h - 1 - nd["y"]
                c.create_text(
                    (nd["x"]+0.5)*cs, (canvas_ndy+1)*cs + 3,
                    text=nd["location_id"],
                    font=("Arial", max(5, cs//5)),
                    fill="#111", tags="label",
                )

        self._update_status()

    def _draw_node_shape(self, nd: dict):
        cs = self.cell_size
        gx, gy = nd["x"], nd["y"]
        cgy = (self.grid_h - 1 - gy) * cs   # canvas Y
        m = cs * 0.18
        zone = nd.get("zone", "")
        outline_color = {"WH": "#1565C0", "ML": "#6A1B9A", "PS": "#2E7D32"}.get(zone, "#444")
        self.canvas.create_oval(
            gx*cs+m, cgy+m, (gx+1)*cs-m, cgy+cs-m,
            fill=nd["color"], outline=outline_color, width=2, tags="node",
        )

    # ── 마우스 이벤트 ──────────────────────────────────────────────────────────

    def _cell_at(self, event) -> tuple | None:
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        gx = int(cx // self.cell_size)
        gy = self.grid_h - 1 - int(cy // self.cell_size)   # flip: Y=0 at bottom
        return (gx, gy) if (0 <= gx < self.grid_w and 0 <= gy < self.grid_h) else None

    def _on_press(self, event):
        cell = self._cell_at(event)
        if not cell: return
        if self.tool_var.get() == "erase":
            self._erase(cell)
        else:
            self.sel_start = cell
            self.sel_cells = {cell}

    def _on_drag(self, event):
        if not self.sel_start: return
        cell = self._cell_at(event)
        tool = self.tool_var.get()
        if tool == "erase":
            if cell: self._erase(cell)
            return
        if not cell: return
        cs = self.cell_size
        min_gx = min(self.sel_start[0], cell[0])
        max_gx = max(self.sel_start[0], cell[0])
        min_gy = min(self.sel_start[1], cell[1])
        max_gy = max(self.sel_start[1], cell[1])
        x0 = min_gx * cs
        x1 = (max_gx + 1) * cs
        y0 = (self.grid_h - 1 - max_gy) * cs   # grid Y큰값 → canvas Y작은값(위)
        y1 = (self.grid_h - min_gy) * cs        # grid Y작은값 → canvas Y큰값(아래)
        if self.sel_rect: self.canvas.delete(self.sel_rect)
        self.sel_rect = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=SEL_COLOR, width=2, dash=(5, 3), tags="sel",
        )
        if tool in ("region", "node"):
            gx0, gy0 = min(self.sel_start[0], cell[0]), min(self.sel_start[1], cell[1])
            gx1, gy1 = max(self.sel_start[0], cell[0]), max(self.sel_start[1], cell[1])
            self.sel_cells = {(gx, gy)
                              for gx in range(gx0, gx1+1)
                              for gy in range(gy0, gy1+1)}

    def _on_release(self, _event):
        if not self.sel_start: return
        tool = self.tool_var.get()
        if self.sel_rect:
            self.canvas.delete(self.sel_rect)
            self.sel_rect = None
        if tool == "region" and self.sel_cells:
            self._dlg_classify_region()
        elif tool == "node" and self.sel_cells:
            self._dlg_classify_node()
        self.sel_start = None
        self.sel_cells = set()

    def _on_wheel(self, event):
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._zoom_in()
        else:
            self._zoom_out()

    # ── 지우개 ─────────────────────────────────────────────────────────────────

    def _erase(self, cell):
        changed = False
        rid = self.cell_map.get(cell)
        if rid:
            self.cell_map.pop(cell)
            if rid in self.regions:
                self.regions[rid]["cells"].discard(cell)
                if not self.regions[rid]["cells"]:
                    del self.regions[rid]
            changed = True
        for nid, nd in list(self.nodes.items()):
            if (nd["x"], nd["y"]) == cell:
                del self.nodes[nid]; changed = True; break
        if changed:
            self._refresh_tree()
            self._redraw()

    # ── 구역 분류 다이얼로그 ────────────────────────────────────────────────────

    def _dlg_classify_region(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("구역 분류")
        dlg.geometry("350x300")
        dlg.resizable(False, False)
        dlg.grab_set(); dlg.transient(self.root)
        self._center(dlg, 350, 300)

        tk.Label(dlg, text="구역 분류 설정",
                 font=("Arial", 11, "bold")).pack(pady=(12, 6))

        pad = {"padx": 16, "pady": 5, "fill": tk.X}

        nf = tk.Frame(dlg); nf.pack(**pad)
        tk.Label(nf, text="이름:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        name_var = tk.StringVar(value=f"구역_{self.next_id}")
        ne = tk.Entry(nf, textvariable=name_var, width=22)
        ne.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ne.select_range(0, tk.END); ne.focus_set()

        tf = tk.Frame(dlg); tf.pack(**pad)
        tk.Label(tf, text="분류 (프리셋):", width=10, anchor=tk.W).pack(side=tk.LEFT)
        type_var = tk.StringVar(value="aisle")
        type_cb = ttk.Combobox(tf, textvariable=type_var,
                               values=list(REGION_PRESETS.keys()),
                               state="readonly", width=20)
        type_cb.pack(side=tk.LEFT)

        uf = tk.Frame(dlg); uf.pack(**pad)
        tk.Label(uf, text="직접 입력:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        cust_var = tk.StringVar()
        tk.Entry(uf, textvariable=cust_var, width=22).pack(side=tk.LEFT)
        tk.Label(uf, text="(비우면 프리셋 사용)", fg="gray",
                 font=("Arial", 7)).pack(side=tk.LEFT, padx=4)

        cf = tk.Frame(dlg); cf.pack(**pad)
        tk.Label(cf, text="색상:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        color_var = tk.StringVar(value=REGION_PRESETS["aisle"][0])
        preview = tk.Label(cf, bg=color_var.get(), width=4, height=1, relief=tk.RAISED)
        preview.pack(side=tk.LEFT, padx=(0, 6))

        def sync(*_):
            t = type_var.get()
            if t in REGION_PRESETS:
                color_var.set(REGION_PRESETS[t][0])
                preview.configure(bg=color_var.get())
        type_cb.bind("<<ComboboxSelected>>", sync)

        def pick():
            c = colorchooser.askcolor(color=color_var.get(), parent=dlg)
            if c[1]: color_var.set(c[1]); preview.configure(bg=c[1])
        tk.Button(cf, text="직접 선택", command=pick).pack(side=tk.LEFT)

        tk.Label(dlg, text=f"선택 셀: {len(self.sel_cells)}개",
                 fg="#555", font=("Arial", 9)).pack(pady=4)

        ok = [False]
        bf = tk.Frame(dlg); bf.pack(pady=8)
        tk.Button(bf, text="확인", width=10, bg=SEL_COLOR, fg="white",
                  command=lambda: ok.__setitem__(0, True) or dlg.destroy()
                  ).pack(side=tk.LEFT, padx=8)
        tk.Button(bf, text="취소", width=10,
                  command=dlg.destroy).pack(side=tk.LEFT, padx=8)
        dlg.bind("<Return>", lambda _: ok.__setitem__(0, True) or dlg.destroy())
        dlg.bind("<Escape>", lambda _: dlg.destroy())
        dlg.wait_window()

        if not ok[0]: return
        name  = name_var.get().strip() or f"구역_{self.next_id}"
        ftype = cust_var.get().strip() or type_var.get()
        self._add_region(name, ftype, color_var.get(), set(self.sel_cells))

    # ── 노드 분류 다이얼로그 (계층형 ID) ──────────────────────────────────────

    def _dlg_classify_node(self):
        # Sort selected cells row-major from top-left:
        # top = larger gy (Y=0 is at bottom), left = smaller gx.
        cells_sorted = sorted(self.sel_cells, key=lambda c: (-c[1], c[0]))
        cell = cells_sorted[0]
        n_cells = len(cells_sorted)
        is_batch = n_cells > 1

        dlg = tk.Toplevel(self.root)
        dlg.title(f"노드 분류 설정{' (일괄)' if is_batch else ''}")
        dlg.geometry("460x640")
        dlg.resizable(False, False)
        dlg.grab_set(); dlg.transient(self.root)
        self._center(dlg, 460, 600)

        # ── ID 미리보기 배너 ──────────────────────────────────────────────────
        banner = tk.Frame(dlg, bg="#EEF4FF", relief=tk.GROOVE, bd=1)
        banner.pack(fill=tk.X, padx=12, pady=(12, 4))

        prefix_text = "자동 생성 ID 시작:" if is_batch else "자동 생성 ID:"
        tk.Label(banner, text=prefix_text, bg="#EEF4FF",
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=10, pady=8)
        id_var = tk.StringVar(value="")
        tk.Label(banner, textvariable=id_var, bg="#EEF4FF",
                 font=("Consolas", 11, "bold"), fg=SEL_COLOR).pack(side=tk.LEFT, padx=4)
        if is_batch:
            tk.Label(banner, text=f"  ×{n_cells}", bg="#EEF4FF",
                     font=("Arial", 9, "bold"), fg="#555").pack(side=tk.LEFT)

        # ── 공통: 제품 + Zone ─────────────────────────────────────────────────
        top = tk.Frame(dlg); top.pack(fill=tk.X, padx=14, pady=3)

        tk.Label(top, text="제품 (Product):", width=16, anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W, pady=4)
        product_var = tk.StringVar(value="A")
        tk.Entry(top, textvariable=product_var, width=8).grid(
            row=0, column=1, sticky=tk.W, pady=4)

        tk.Label(top, text="구역 (Zone):", width=16, anchor=tk.W).grid(
            row=1, column=0, sticky=tk.W, pady=4)
        zone_var = tk.StringVar(value="WH")
        zf = tk.Frame(top); zf.grid(row=1, column=1, sticky=tk.W)
        for val, lbl in [("WH", "WH  창고"), ("ML", "ML  제조라인"), ("PS", "PS  주차장")]:
            tk.Radiobutton(zf, text=lbl, variable=zone_var, value=val,
                           font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 10))

        sep = ttk.Separator(dlg, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=14, pady=6)

        # ── WH 전용 폼 ────────────────────────────────────────────────────────
        wh_frame = tk.LabelFrame(dlg, text="WH 창고 위치 정보", font=("Arial", 9, "bold"),
                                 padx=10, pady=6)

        wf = tk.Frame(wh_frame)
        wf.pack(fill=tk.X)

        # Section
        tk.Label(wf, text="섹션 (Section):", width=16, anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W, pady=4)
        wh_sec_var = tk.StringVar(value="F")
        sf = tk.Frame(wf); sf.grid(row=0, column=1, sticky=tk.W)
        for val, lbl in [("F", "F  전방 데포 (Front)"), ("B", "B  후방 데포 (Back)")]:
            tk.Radiobutton(sf, text=lbl, variable=wh_sec_var, value=val,
                           font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 8))

        # Part
        tk.Label(wf, text="부품 (Part):", width=16, anchor=tk.W).grid(
            row=1, column=0, sticky=tk.W, pady=4)
        wh_part_var = tk.StringVar(value="a")
        pf = tk.Frame(wf); pf.grid(row=1, column=1, sticky=tk.W)
        tk.Entry(pf, textvariable=wh_part_var, width=6).pack(side=tk.LEFT)
        tk.Label(pf, text="  예: a, b, c, d, e …",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT)

        # Rack type
        tk.Label(wf, text="랙 타입:", width=16, anchor=tk.W).grid(
            row=2, column=0, sticky=tk.W, pady=4)
        wh_rack_var = tk.StringVar(value="O")
        rf = tk.Frame(wf); rf.grid(row=2, column=1, sticky=tk.W)
        rack_cb = ttk.Combobox(rf, textvariable=wh_rack_var,
                               values=list(RACK_TYPES.keys()),
                               state="readonly", width=5)
        rack_cb.pack(side=tk.LEFT, padx=(0, 6))
        rack_desc_var = tk.StringVar(value=RACK_TYPES["O"][1])
        tk.Label(rf, textvariable=rack_desc_var,
                 fg="#555", font=("Arial", 8), wraplength=220,
                 justify=tk.LEFT).pack(side=tk.LEFT)

        def on_rack(*_):
            rack_desc_var.set(RACK_TYPES.get(wh_rack_var.get(), ("", ""))[1])
        rack_cb.bind("<<ComboboxSelected>>", on_rack)

        # Slot
        tk.Label(wf, text="슬롯 (Slot):", width=16, anchor=tk.W).grid(
            row=3, column=0, sticky=tk.W, pady=4)
        wh_slot_var = tk.IntVar(value=1)
        slotf = tk.Frame(wf); slotf.grid(row=3, column=1, sticky=tk.W)
        tk.Spinbox(slotf, from_=1, to=99, textvariable=wh_slot_var, width=5).pack(side=tk.LEFT)
        tk.Label(slotf, text="  (1, 2, 3 …)",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT)

        # ── PS 전용 폼 ────────────────────────────────────────────────────────
        ps_frame = tk.LabelFrame(dlg, text="PS AMR 주차장 정보", font=("Arial", 9, "bold"),
                                 padx=10, pady=6)

        psf = tk.Frame(ps_frame)
        psf.pack(fill=tk.X)

        # AMR 타입
        tk.Label(psf, text="AMR 타입:", width=16, anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W, pady=4)
        ps_amr_var = tk.StringVar(value="TypeA")
        amrf = tk.Frame(psf); amrf.grid(row=0, column=1, sticky=tk.W)
        amr_cb = ttk.Combobox(amrf, textvariable=ps_amr_var,
                              values=list(AMR_TYPES.keys()),
                              state="readonly", width=10)
        amr_cb.pack(side=tk.LEFT, padx=(0, 6))
        amr_desc_var = tk.StringVar(value=AMR_TYPES["TypeA"][1])
        tk.Label(amrf, textvariable=amr_desc_var,
                 fg="#555", font=("Arial", 8)).pack(side=tk.LEFT)

        def on_amr(*_):
            amr_desc_var.set(AMR_TYPES.get(ps_amr_var.get(), ("", ""))[1])
        amr_cb.bind("<<ComboboxSelected>>", on_amr)

        # PS 슬롯
        tk.Label(psf, text="슬롯 (Slot):", width=16, anchor=tk.W).grid(
            row=1, column=0, sticky=tk.W, pady=4)
        ps_slot_var = tk.IntVar(value=1)
        psslotf = tk.Frame(psf); psslotf.grid(row=1, column=1, sticky=tk.W)
        tk.Spinbox(psslotf, from_=1, to=99, textvariable=ps_slot_var, width=5).pack(side=tk.LEFT)
        tk.Label(psslotf, text="  (1, 2, 3 …)",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT)

        # ── ML 전용 폼 ────────────────────────────────────────────────────────
        ml_frame = tk.LabelFrame(dlg, text="ML 제조라인 노드 정보", font=("Arial", 9, "bold"),
                                 padx=10, pady=6)

        mf = tk.Frame(ml_frame)
        mf.pack(fill=tk.X)

        # Station
        tk.Label(mf, text="스테이션 (Station):", width=18, anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W, pady=4)
        ml_station_var = tk.StringVar(value="a")
        stf = tk.Frame(mf); stf.grid(row=0, column=1, sticky=tk.W)
        tk.Entry(stf, textvariable=ml_station_var, width=6).pack(side=tk.LEFT)
        tk.Label(stf, text="  예: a, b, c …",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT)

        # Node type
        tk.Label(mf, text="노드 타입:", width=18, anchor=tk.W).grid(
            row=1, column=0, sticky=tk.W, pady=4)
        ml_nt_var = tk.StringVar(value="R")
        ntf = tk.Frame(mf); ntf.grid(row=1, column=1, sticky=tk.W)
        nt_cb = ttk.Combobox(ntf, textvariable=ml_nt_var,
                             values=list(ML_NODE_TYPES.keys()),
                             state="readonly", width=5)
        nt_cb.pack(side=tk.LEFT, padx=(0, 6))
        nt_desc_var = tk.StringVar(value=ML_NODE_TYPES["R"][1])
        tk.Label(ntf, textvariable=nt_desc_var,
                 fg="#555", font=("Arial", 8), wraplength=220,
                 justify=tk.LEFT).pack(side=tk.LEFT)

        def on_nt(*_):
            nt_desc_var.set(ML_NODE_TYPES.get(ml_nt_var.get(), ("", ""))[1])
        nt_cb.bind("<<ComboboxSelected>>", on_nt)

        # Index
        tk.Label(mf, text="인덱스 (Index):", width=18, anchor=tk.W).grid(
            row=2, column=0, sticky=tk.W, pady=4)
        ml_idx_var = tk.IntVar(value=1)
        nif = tk.Frame(mf); nif.grid(row=2, column=1, sticky=tk.W)
        tk.Spinbox(nif, from_=1, to=99, textvariable=ml_idx_var, width=5).pack(side=tk.LEFT)
        tk.Label(nif, text="  → ID: R1, R2, P1 …",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT)

        # ── 공통 하단 ─────────────────────────────────────────────────────────
        sep2 = ttk.Separator(dlg, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, padx=14, pady=6)

        bot = tk.Frame(dlg); bot.pack(fill=tk.X, padx=14)

        tk.Label(bot, text="접근 노드 ID:", width=16, anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W, pady=3)
        access_var = tk.StringVar()
        anf = tk.Frame(bot); anf.grid(row=0, column=1, sticky=tk.W)
        tk.Entry(anf, textvariable=access_var, width=12).pack(side=tk.LEFT)
        tk.Label(anf, text="  예: N_101 (선택사항)",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT)

        tk.Label(bot, text="점유 (Occupied):", width=16, anchor=tk.W).grid(
            row=1, column=0, sticky=tk.W, pady=3)
        occ_var = tk.IntVar(value=0)
        ttk.Combobox(bot, textvariable=occ_var, values=[0, 1],
                     state="readonly", width=6).grid(row=1, column=1, sticky=tk.W, pady=3)

        if is_batch:
            tk.Label(
                dlg,
                text=(
                    f"드래그 영역: {n_cells}개 셀  "
                    f"(시작 ({cell[0]}, {cell[1]}) → 왼쪽 위에서 오른쪽 아래로 슬롯 자동 증가)"
                ),
                fg="#1565C0", font=("Arial", 9, "bold"),
            ).pack(pady=2)
        else:
            tk.Label(dlg, text=f"배치 위치: ({cell[0]}, {cell[1]})",
                     fg="#444", font=("Arial", 9)).pack(pady=2)

        # ── ID 자동 생성 로직 ─────────────────────────────────────────────────
        def update_id(*_):
            p = product_var.get().strip() or "?"
            z = zone_var.get()
            if z == "WH":
                parts = [p, "WH",
                         wh_sec_var.get(),
                         wh_part_var.get().strip() or "?",
                         wh_rack_var.get(),
                         str(wh_slot_var.get())]
            elif z == "ML":
                parts = [p, "ML",
                         ml_station_var.get().strip() or "?",
                         ml_nt_var.get() + str(ml_idx_var.get())]
            else:
                parts = [p, "PS",
                         ps_amr_var.get().strip() or "?",
                         str(ps_slot_var.get())]
            id_var.set("_".join(parts))

        def on_zone_switch(*_):
            z = zone_var.get()
            wh_frame.pack_forget()
            ml_frame.pack_forget()
            ps_frame.pack_forget()
            if z == "WH":
                wh_frame.pack(fill=tk.X, padx=14, pady=(0, 4))
            elif z == "ML":
                ml_frame.pack(fill=tk.X, padx=14, pady=(0, 4))
            else:
                ps_frame.pack(fill=tk.X, padx=14, pady=(0, 4))
            update_id()

        zone_var.trace_add("write", on_zone_switch)
        for v in [product_var, wh_sec_var, wh_part_var, wh_rack_var,
                  wh_slot_var, ml_station_var, ml_nt_var, ml_idx_var,
                  ps_amr_var, ps_slot_var]:
            v.trace_add("write", update_id)

        # 초기 표시
        wh_frame.pack(fill=tk.X, padx=14, pady=(0, 4))
        update_id()

        # ── 버튼 ──────────────────────────────────────────────────────────────
        ok = [False]
        bf = tk.Frame(dlg); bf.pack(pady=8)
        tk.Button(bf, text="확인", width=10, bg=SEL_COLOR, fg="white",
                  command=lambda: ok.__setitem__(0, True) or dlg.destroy()
                  ).pack(side=tk.LEFT, padx=8)
        tk.Button(bf, text="취소", width=10,
                  command=dlg.destroy).pack(side=tk.LEFT, padx=8)
        dlg.bind("<Return>", lambda _: ok.__setitem__(0, True) or dlg.destroy())
        dlg.bind("<Escape>", lambda _: dlg.destroy())
        dlg.wait_window()

        if not ok[0]: return

        # 색상 자동 결정
        z = zone_var.get()
        if z == "WH":
            color = ZONE_COLORS["WH_F"] if wh_sec_var.get() == "F" else ZONE_COLORS["WH_B"]
        elif z == "ML":
            color = ZONE_COLORS["ML"]
        else:
            color = ZONE_COLORS["PS"]

        # 시작 슬롯 + 공통 필드를 한 번만 읽고, 셀마다 슬롯만 증가.
        if z == "WH":
            start_slot = wh_slot_var.get()
        elif z == "ML":
            start_slot = ml_idx_var.get()
        else:
            start_slot = ps_slot_var.get()

        product_val = product_var.get().strip()
        access_val = access_var.get().strip() or None
        occ_val = occ_var.get()

        for i, batch_cell in enumerate(cells_sorted):
            slot_val = start_slot + i

            # 슬롯에 맞춘 location_id 재생성 (update_id 함수와 동일한 규칙).
            if z == "WH":
                loc_id = "_".join([
                    product_val or "?", "WH",
                    wh_sec_var.get(),
                    wh_part_var.get().strip() or "?",
                    wh_rack_var.get(),
                    str(slot_val),
                ])
            elif z == "ML":
                loc_id = "_".join([
                    product_val or "?", "ML",
                    ml_station_var.get().strip() or "?",
                    ml_nt_var.get() + str(slot_val),
                ])
            else:
                loc_id = "_".join([
                    product_val or "?", "PS",
                    ps_amr_var.get().strip() or "?",
                    str(slot_val),
                ])

            node_data: dict = {
                "location_id":    loc_id,
                "product":        product_val,
                "zone":           z,
                "access_node_id": access_val,
                "occupied":       occ_val,
                "color":          color,
                "x":              batch_cell[0],
                "y":              batch_cell[1],
            }
            if z == "WH":
                rack_code = wh_rack_var.get()
                node_data.update({
                    "section":         wh_sec_var.get(),
                    "section_label":   "Front" if wh_sec_var.get() == "F" else "Back",
                    "part":            wh_part_var.get().strip(),
                    "rack_type":       rack_code,
                    "rack_type_label": RACK_TYPES.get(rack_code, (rack_code, ""))[0],
                    "slot":            slot_val,
                })
            elif z == "ML":
                nt_code = ml_nt_var.get()
                node_data.update({
                    "part":             ml_station_var.get().strip(),
                    "rack_type":        nt_code,
                    "rack_type_label":  ML_NODE_TYPES.get(nt_code, (nt_code, ""))[0],
                    "slot":             slot_val,
                })
            else:
                node_data.update({
                    "amr_type": ps_amr_var.get().strip(),
                    "slot":     slot_val,
                })

            self._add_node(node_data)

    # ── 데이터 추가 ─────────────────────────────────────────────────────────────

    def _add_region(self, name: str, rtype: str, color: str, cells: set):
        rid = f"region_{self.next_id}"
        self.next_id += 1
        for cell in cells:
            old = self.cell_map.get(cell)
            if old and old in self.regions:
                self.regions[old]["cells"].discard(cell)
        self.regions[rid] = {
            "id": rid, "name": name, "type": rtype, "color": color, "cells": cells,
        }
        for cell in cells:
            self.cell_map[cell] = rid
        self._refresh_tree()
        self._redraw()

    def _add_node(self, data: dict):
        nid = f"node_{self.next_id}"
        self.next_id += 1
        for eid, nd in list(self.nodes.items()):
            if nd["x"] == data["x"] and nd["y"] == data["y"]:
                del self.nodes[eid]; break
        data["_nid"] = nid
        self.nodes[nid] = data
        self._refresh_tree()
        self._redraw()

    # ── 사이드바 갱신 ──────────────────────────────────────────────────────────

    def _refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for rid, r in self.regions.items():
            tag = f"c_{rid}"
            self.tree.tag_configure(tag, background=r["color"])
            self.tree.insert("", tk.END, iid=rid, tags=(tag,),
                             values=(r["name"], r["type"], f"{len(r['cells'])}셀"))
        for nid, nd in self.nodes.items():
            tag = f"c_{nid}"
            self.tree.tag_configure(tag, background=nd["color"])
            self.tree.insert("", tk.END, iid=nid, tags=(tag,),
                             values=(nd["location_id"], nd.get("zone", "?"),
                                     f"({nd['x']},{nd['y']})"))

    # ── 편집 ───────────────────────────────────────────────────────────────────

    def _edit_selected(self):
        sel = self.tree.selection()
        if not sel: return
        iid = sel[0]
        if iid in self.regions:
            self._dlg_edit_region(iid)
        elif iid in self.nodes:
            self._dlg_edit_node(iid)

    def _dlg_edit_region(self, rid):
        r = self.regions[rid]
        dlg = tk.Toplevel(self.root)
        dlg.title("구역 편집"); dlg.geometry("320x210")
        dlg.grab_set(); dlg.transient(self.root)
        self._center(dlg, 320, 210)

        vars_ = {}
        for i, (lbl, key) in enumerate([("이름:", "name"), ("분류:", "type")]):
            tk.Label(dlg, text=lbl, width=8, anchor=tk.W).grid(
                row=i, column=0, padx=12, pady=8, sticky=tk.W)
            v = tk.StringVar(value=r[key])
            tk.Entry(dlg, textvariable=v, width=24).grid(row=i, column=1, padx=8, pady=8)
            vars_[key] = v

        tk.Label(dlg, text="색상:", width=8, anchor=tk.W).grid(
            row=2, column=0, padx=12, pady=8, sticky=tk.W)
        cf = tk.Frame(dlg); cf.grid(row=2, column=1, sticky=tk.W)
        color_v = tk.StringVar(value=r["color"])
        prev = tk.Label(cf, bg=color_v.get(), width=3, height=1, relief=tk.RAISED)
        prev.pack(side=tk.LEFT, padx=(0, 6))
        def pick():
            c = colorchooser.askcolor(color=color_v.get(), parent=dlg)
            if c[1]: color_v.set(c[1]); prev.configure(bg=c[1])
        tk.Button(cf, text="선택", command=pick).pack(side=tk.LEFT)

        def save():
            for key, v in vars_.items(): r[key] = v.get().strip() or r[key]
            r["color"] = color_v.get()
            self._refresh_tree(); self._redraw(); dlg.destroy()
        bf = tk.Frame(dlg); bf.grid(row=3, column=0, columnspan=2, pady=10)
        tk.Button(bf, text="저장", width=8, bg=SEL_COLOR, fg="white",
                  command=save).pack(side=tk.LEFT, padx=6)
        tk.Button(bf, text="취소", width=8, command=dlg.destroy).pack(side=tk.LEFT, padx=6)
        dlg.bind("<Return>", lambda _: save())

    def _dlg_edit_node(self, nid):
        nd = self.nodes[nid]
        dlg = tk.Toplevel(self.root)
        dlg.title("노드 편집"); dlg.geometry("400x460")
        dlg.grab_set(); dlg.transient(self.root)
        self._center(dlg, 400, 460)

        tk.Label(dlg, text=nd["location_id"],
                 font=("Consolas", 11, "bold"), fg=SEL_COLOR).pack(pady=(10, 2))
        tk.Label(dlg, text=f"Zone: {nd.get('zone','')}",
                 fg="#555", font=("Arial", 9)).pack(pady=(0, 8))

        if nd.get("zone") == "WH":
            edit_keys = [("섹션 (F/B)",  "section"),
                         ("부품",         "part"),
                         ("랙 타입",      "rack_type"),
                         ("슬롯",         "slot")]
        elif nd.get("zone") == "ML":
            edit_keys = [("파트 (Part)",   "part"),
                         ("랙 타입",       "rack_type"),
                         ("슬롯",          "slot")]
        elif nd.get("zone") == "PS":
            edit_keys = [("AMR 타입",      "amr_type"),
                         ("슬롯",          "slot")]
        else:
            edit_keys = []
        edit_keys += [
            ("좌표 X",         "x"),
            ("좌표 Y",         "y"),
            ("접근 노드 ID",   "access_node_id"),
            ("점유",           "occupied"),
        ]

        frame = tk.Frame(dlg); frame.pack(fill=tk.BOTH, padx=14)
        vars_: dict[str, tk.StringVar] = {}
        for i, (lbl, key) in enumerate(edit_keys):
            tk.Label(frame, text=f"{lbl}:", width=16, anchor=tk.W).grid(
                row=i, column=0, pady=5, sticky=tk.W)
            v = tk.StringVar(value=str(nd.get(key, "")))
            tk.Entry(frame, textvariable=v, width=20).grid(row=i, column=1, pady=5, sticky=tk.W)
            vars_[key] = v

        tk.Label(
            dlg,
            text=f"※ 그리드 범위: X 0~{self.grid_w - 1}, Y 0~{self.grid_h - 1}",
            fg="#888", font=("Arial", 8),
        ).pack(pady=(2, 0))

        def save():
            # 좌표 검증 + 다른 노드와 충돌 검사
            try:
                new_x = int(vars_["x"].get().strip())
                new_y = int(vars_["y"].get().strip())
            except ValueError:
                messagebox.showerror("오류", "좌표 X, Y는 정수여야 합니다.", parent=dlg)
                return
            if not (0 <= new_x < self.grid_w and 0 <= new_y < self.grid_h):
                messagebox.showerror(
                    "오류",
                    f"좌표는 X 0~{self.grid_w - 1}, Y 0~{self.grid_h - 1} 범위여야 합니다.",
                    parent=dlg,
                )
                return
            for other_nid, other in self.nodes.items():
                if other_nid == nid:
                    continue
                if other["x"] == new_x and other["y"] == new_y:
                    messagebox.showerror(
                        "오류",
                        f"({new_x}, {new_y})에 이미 다른 노드 "
                        f"'{other['location_id']}'가 있습니다.",
                        parent=dlg,
                    )
                    return

            nd["x"] = new_x
            nd["y"] = new_y
            for key, v in vars_.items():
                if key in ("x", "y"):
                    continue
                val = v.get().strip()
                if key in ("slot", "occupied"):
                    try: nd[key] = int(val)
                    except ValueError: pass
                else:
                    nd[key] = val if val else nd.get(key)
            self._refresh_tree(); self._redraw(); dlg.destroy()

        bf = tk.Frame(dlg); bf.pack(pady=10)
        tk.Button(bf, text="저장", width=8, bg=SEL_COLOR, fg="white",
                  command=save).pack(side=tk.LEFT, padx=6)
        tk.Button(bf, text="취소", width=8, command=dlg.destroy).pack(side=tk.LEFT, padx=6)
        dlg.bind("<Return>", lambda _: save())

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel: return
        iid = sel[0]
        label = (self.regions.get(iid, {}).get("name")
                 or self.nodes.get(iid, {}).get("location_id", iid))
        if not messagebox.askyesno("삭제 확인", f"'{label}'을(를) 삭제할까요?"): return
        if iid in self.regions:
            for cell in self.regions[iid]["cells"]:
                if self.cell_map.get(cell) == iid: del self.cell_map[cell]
            del self.regions[iid]
        elif iid in self.nodes:
            del self.nodes[iid]
        self._refresh_tree()
        self._redraw()

    # ── 확대/축소 ──────────────────────────────────────────────────────────────

    def _zoom_in(self):
        self.cell_size = min(self.cell_size + 4, MAX_CELL); self._redraw()
    def _zoom_out(self):
        self.cell_size = max(self.cell_size - 4, MIN_CELL); self._redraw()

    # ── 그리드 크기 설정 ───────────────────────────────────────────────────────

    def _dlg_grid_size(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("그리드 크기 설정"); dlg.geometry("270x150")
        dlg.grab_set(); dlg.transient(self.root)
        self._center(dlg, 270, 150)
        tk.Label(dlg, text="너비:").grid(row=0, column=0, padx=14, pady=10, sticky=tk.W)
        w_var = tk.IntVar(value=self.grid_w)
        tk.Spinbox(dlg, from_=5, to=500, textvariable=w_var, width=8).grid(row=0, column=1, padx=10)
        tk.Label(dlg, text="높이:").grid(row=1, column=0, padx=14, pady=6, sticky=tk.W)
        h_var = tk.IntVar(value=self.grid_h)
        tk.Spinbox(dlg, from_=5, to=500, textvariable=h_var, width=8).grid(row=1, column=1, padx=10)

        def apply():
            self.grid_w, self.grid_h = w_var.get(), h_var.get()
            for cell in list(self.cell_map):
                if cell[0] >= self.grid_w or cell[1] >= self.grid_h:
                    rid = self.cell_map.pop(cell)
                    if rid in self.regions: self.regions[rid]["cells"].discard(cell)
            self._redraw(); dlg.destroy()
        bf = tk.Frame(dlg); bf.grid(row=2, column=0, columnspan=2, pady=12)
        tk.Button(bf, text="적용", width=8, bg=SEL_COLOR, fg="white",
                  command=apply).pack(side=tk.LEFT, padx=6)
        tk.Button(bf, text="취소", width=8, command=dlg.destroy).pack(side=tk.LEFT, padx=6)

    # ── 새 맵 ──────────────────────────────────────────────────────────────────

    def _new_map(self):
        if self.regions or self.nodes:
            if not messagebox.askyesno("새로 만들기",
                                       "현재 작업을 모두 지우고 새로 시작할까요?"): return
        self.regions.clear(); self.cell_map.clear()
        self.nodes.clear(); self.next_id = 1
        self._refresh_tree(); self._redraw()

    # ── JSON 열기 ──────────────────────────────────────────────────────────────

    def _open_json(self):
        if self.regions or self.nodes:
            if not messagebox.askyesno("JSON 열기",
                                       "현재 작업을 지우고 파일을 불러올까요?"): return
        path = filedialog.askopenfilename(
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")],
            title="JSON 열기",
        )
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("오류", f"파일 읽기 실패:\n{e}"); return

        self.regions.clear(); self.cell_map.clear()
        self.nodes.clear(); self.next_id = 1

        # 그리드 크기
        meta = data.get("metadata", {})
        grid = meta.get("grid", {})
        self.grid_w = grid.get("width",  DEFAULT_W)
        self.grid_h = grid.get("height", DEFAULT_H)

        # 구역(Region) 복원
        for region in data.get("regions", []):
            rid = f"region_{self.next_id}"; self.next_id += 1
            cells = set(tuple(c) for c in region.get("cells", []))
            self.regions[rid] = {
                "id": rid, "name": region.get("name", ""),
                "type": region.get("type", ""), "color": region.get("color", "#E0E0E0"),
                "cells": cells,
            }
            for cell in cells:
                self.cell_map[cell] = rid

        # WH 노드 복원
        for loc in data.get("wh_locations", []):
            nid = f"node_{self.next_id}"; self.next_id += 1
            coords = loc.get("coordinates", {})
            sec = loc.get("section", "F")
            color = ZONE_COLORS["WH_F"] if sec == "F" else ZONE_COLORS["WH_B"]
            self.nodes[nid] = {
                "_nid": nid, "zone": "WH", "color": color,
                "x": coords.get("x", 0), "y": coords.get("y", 0),
                "location_id":    loc.get("location_id", ""),
                "product":        loc.get("product", ""),
                "access_node_id": loc.get("access_node_id"),
                "occupied":       loc.get("occupied", 0),
                "section":        sec,
                "section_label":  loc.get("section_label", ""),
                "part":           loc.get("part", ""),
                "rack_type":      loc.get("rack_type", ""),
                "rack_type_label": loc.get("rack_type_label", ""),
                "slot":           loc.get("slot", 1),
            }

        # ML 노드 복원 (구/신 스키마 모두 지원)
        for node in data.get("ml_nodes", []):
            nid = f"node_{self.next_id}"; self.next_id += 1
            coords = node.get("coordinates", {})
            self.nodes[nid] = {
                "_nid": nid, "zone": "ML", "color": ZONE_COLORS["ML"],
                "x": coords.get("x", 0), "y": coords.get("y", 0),
                "location_id":    node.get("location_id", ""),
                "product":        node.get("product", ""),
                "access_node_id": node.get("access_node_id"),
                "occupied":       node.get("occupied", 0),
                "part":           node.get("part") or node.get("station", ""),
                "rack_type":      node.get("rack_type") or node.get("node_type", ""),
                "rack_type_label": node.get("rack_type_label") or node.get("node_type_label", ""),
                "slot":           node.get("slot") or node.get("node_index", 1),
            }

        # PS 노드 복원
        for node in data.get("ps_nodes", []):
            nid = f"node_{self.next_id}"; self.next_id += 1
            coords = node.get("coordinates", {})
            self.nodes[nid] = {
                "_nid": nid, "zone": "PS", "color": ZONE_COLORS["PS"],
                "x": coords.get("x", 0), "y": coords.get("y", 0),
                "location_id":    node.get("location_id", ""),
                "product":        node.get("product", ""),
                "access_node_id": node.get("access_node_id"),
                "occupied":       node.get("occupied", 0),
                "amr_type":       node.get("amr_type", ""),
                "slot":           node.get("slot", 1),
            }

        self._refresh_tree(); self._redraw()
        messagebox.showinfo("열기 완료",
            f"구역 {len(self.regions)}개 / "
            f"WH {sum(1 for n in self.nodes.values() if n.get('zone')=='WH')}개 / "
            f"ML {sum(1 for n in self.nodes.values() if n.get('zone')=='ML')}개 / "
            f"PS {sum(1 for n in self.nodes.values() if n.get('zone')=='PS')}개 불러왔습니다.\n{path}")

    # ── JSON 내보내기 ──────────────────────────────────────────────────────────

    def _export_json(self):
        if not self.regions and not self.nodes:
            messagebox.showwarning("내보내기", "내보낼 데이터가 없습니다."); return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")],
            initialfile=f"map_{datetime.now().strftime('%y%m%d_%H%M%S')}.json",
            title="JSON 내보내기",
        )
        if not path: return

        # ── 구역 ──────────────────────────────────────────────────────────────
        regions_out = []
        for rid, r in self.regions.items():
            cells = sorted(r["cells"])
            xs, ys = [c[0] for c in cells], [c[1] for c in cells]
            regions_out.append({
                "id": rid, "name": r["name"], "type": r["type"], "color": r["color"],
                "cell_count": len(cells),
                "bounding_box": {
                    "x_min": min(xs), "y_min": min(ys),
                    "x_max": max(xs), "y_max": max(ys),
                    "width":  max(xs) - min(xs) + 1,
                    "height": max(ys) - min(ys) + 1,
                },
                "cells": [[x, y] for x, y in cells],
            })

        # ── 노드: WH / ML / PS 분리 ───────────────────────────────────────────
        wh_locations: list[dict] = []
        ml_nodes:     list[dict] = []
        ps_nodes:     list[dict] = []

        for nd in self.nodes.values():
            base = {
                "location_id":    nd["location_id"],
                "product":        nd.get("product", ""),
                "zone":           nd.get("zone", ""),
                "coordinates":    {"x": nd["x"], "y": nd["y"]},
                "access_node_id": nd.get("access_node_id"),
                "occupied":       nd.get("occupied", 0),
            }
            if nd.get("zone") == "WH":
                base.update({
                    "section":         nd.get("section", ""),
                    "section_label":   nd.get("section_label", ""),
                    "part":            nd.get("part", ""),
                    "rack_type":       nd.get("rack_type", ""),
                    "rack_type_label": nd.get("rack_type_label", ""),
                    "slot":            nd.get("slot", 0),
                })
                wh_locations.append(base)
            elif nd.get("zone") == "ML":
                base.update({
                    "part":            nd.get("part", ""),
                    "rack_type":       nd.get("rack_type", ""),
                    "rack_type_label": nd.get("rack_type_label", ""),
                    "slot":            nd.get("slot", 0),
                })
                ml_nodes.append(base)
            elif nd.get("zone") == "PS":
                base.update({
                    "amr_type": nd.get("amr_type", ""),
                    "slot":     nd.get("slot", 0),
                })
                ps_nodes.append(base)
            else:
                wh_locations.append(base)

        data = {
            "metadata": {
                "created":         datetime.now().isoformat(),
                "grid":            {"width": self.grid_w, "height": self.grid_h},
                "cell_size_unit":  1,
                "id_format": {
                    "wh": "{product}_{zone}_{section}_{part}_{rack_type}_{slot}",
                    "ml": "{product}_{zone}_{part}_{rack_type}{slot}",
                    "ps": "{product}_{zone}_{amr_type}_{slot}",
                },
                "rack_type_legend": {
                    k: v[0] for k, v in RACK_TYPES.items()
                },
                "ml_node_type_legend": {
                    k: v[0] for k, v in ML_NODE_TYPES.items()
                },
                "total_regions":  len(regions_out),
                "total_wh_nodes": len(wh_locations),
                "total_ml_nodes": len(ml_nodes),
                "total_ps_nodes": len(ps_nodes),
            },
            "regions":      regions_out,
            "wh_locations": wh_locations,
            "ml_nodes":     ml_nodes,
            "ps_nodes":     ps_nodes,
            "cell_map": {
                f"{gx},{gy}": {
                    "region_id": rid,
                    "type":      self.regions[rid]["type"],
                    "name":      self.regions[rid]["name"],
                }
                for (gx, gy), rid in self.cell_map.items()
                if rid in self.regions
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        messagebox.showinfo(
            "내보내기 완료",
            f"저장 완료!\n\n"
            f"구역(Region): {len(regions_out)}개\n"
            f"WH 위치:      {len(wh_locations)}개\n"
            f"ML 노드:      {len(ml_nodes)}개\n"
            f"PS 노드:      {len(ps_nodes)}개\n\n{path}",
        )

    # ── 유틸 ───────────────────────────────────────────────────────────────────

    def _center(self, win: tk.Toplevel, w: int, h: int):
        win.update_idletasks()
        px = self.root.winfo_x() + self.root.winfo_width()  // 2 - w // 2
        py = self.root.winfo_y() + self.root.winfo_height() // 2 - h // 2
        win.geometry(f"+{px}+{py}")

    def _update_status(self):
        tool_names = {"region": "구역 선택 (드래그)",
                      "node":   "노드 배치 (클릭)",
                      "erase":  "지우개"}
        t = self.tool_var.get() if hasattr(self, "tool_var") else "region"
        n_wh = sum(1 for n in self.nodes.values() if n.get("zone") == "WH")
        n_ml = sum(1 for n in self.nodes.values() if n.get("zone") == "ML")
        n_ps = sum(1 for n in self.nodes.values() if n.get("zone") == "PS")
        self.status_var.set(
            f"도구: {tool_names.get(t, t)}  |  "
            f"그리드: {self.grid_w}×{self.grid_h}  |  "
            f"구역: {len(self.regions)}개  |  "
            f"WH: {n_wh}개  |  ML: {n_ml}개  |  PS: {n_ps}개"
        )


# ── 진입점 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    MapEditor(root)
    root.mainloop()
