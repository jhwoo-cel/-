from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "simulation_log"


def find_latest_kpi(log_dir: Path = LOG_DIR) -> Path:
    candidates = sorted(log_dir.glob("kpi_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"KPI JSON 파일을 찾을 수 없습니다: {log_dir}")
    return candidates[-1]


def load_kpi(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def as_number(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_num(value, digits: int = 4) -> str:
    number = as_number(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def fmt_percent(value) -> str:
    return f"{as_number(value):.4f}".rstrip("0").rstrip(".")


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def get_dict(data: dict, key: str) -> dict:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def amr_records(kpi: dict) -> list[dict]:
    records = []
    work_utilization = get_dict(kpi, "이동로봇별_작업가동률")
    for amr_id, record in sorted(get_dict(kpi, "이동로봇별_가동률").items()):
        if not isinstance(record, dict):
            continue
        work_record = work_utilization.get(amr_id, {})
        if not isinstance(work_record, dict):
            work_record = {}
        records.append(
            {
                "id": amr_id,
                "utilization": as_number(record.get("가동률_퍼센트")),
                "work_utilization": as_number(work_record.get("작업가동률_퍼센트")),
                "work_time": as_number(work_record.get("작업가동시간")),
                "move_time": as_number(record.get("이동시간")),
                "distance": as_number(record.get("이동거리")),
                "assignments": as_number(record.get("배정횟수")),
                "completed_moves": as_number(record.get("이동완료횟수")),
            }
        )
    return records


def depot_records(kpi: dict) -> list[dict]:
    records = []
    for group_name, record in sorted(get_dict(kpi, "파트별_자재데포_공급지표").items()):
        if not isinstance(record, dict):
            continue
        records.append(
            {
                "name": group_name,
                "part": record.get("부품"),
                "transfers": as_number(record.get("이송횟수")),
                "wait_time": as_number(record.get("평균_대기시간")),
                "supply_time": as_number(record.get("평균_공급시간")),
                "depots": list(record.get("포함_자재데포") or []),
            }
        )
    return records


def unfinished_depot_records(kpi: dict) -> list[dict]:
    records = []
    source = get_dict(kpi, "데포별_병목지표") or get_dict(kpi, "데포별_미완료_진입지표")
    for depot_id, record in sorted(source.items()):
        if not isinstance(record, dict):
            continue
        records.append(
            {
                "id": depot_id,
                "type": record.get("데포종류"),
                "product": record.get("제품"),
                "part": record.get("부품"),
                "total": as_number(
                    record.get("누적_병목횟수", record.get("누적_미완료_진입수"))
                ),
            }
        )
    return records


def average_amr_utilization(kpi: dict) -> float:
    values = [record["work_utilization"] for record in amr_records(kpi)]
    return sum(values) / len(values) if values else 0.0


def active_depot_count(kpi: dict) -> int:
    depots = set()
    for record in depot_records(kpi):
        depots.update(record["depots"])
    return len(depots)


def bottleneck_part(kpi: dict) -> tuple[str, float]:
    records = depot_records(kpi)
    if not records:
        return "-", 0.0
    record = max(records, key=lambda item: item["supply_time"])
    return record["name"], record["supply_time"]


def render_summary_rows(kpi: dict) -> str:
    bottleneck_name, bottleneck_supply = bottleneck_part(kpi)
    rows = [
        ("시뮬레이션 종료시간", kpi.get("시뮬레이션_종료시간"), "sec"),
        ("총 작업배정횟수", kpi.get("총_작업배정횟수"), "회"),
        ("총 자재데포 -> 공정데포 이송횟수", kpi.get("총_자재데포_공정데포_이송횟수"), "회"),
        (
            "총 공정데포 병목횟수",
            kpi.get("총_공정데포_병목횟수", kpi.get("총_공정데포_미완료_진입횟수")),
            "회",
        ),
        (
            "총 회수데포 병목횟수",
            kpi.get("총_회수데포_병목횟수", kpi.get("총_회수데포_미완료_진입횟수")),
            "회",
        ),
        ("평균 대기시간", kpi.get("평균_대기시간"), "sec"),
        ("평균 공급시간", kpi.get("평균_공급시간"), "sec"),
        ("AMR 평균 작업가동률", average_amr_utilization(kpi), "%"),
        ("활성 자재데포 수", active_depot_count(kpi), "개"),
        (f"최대 평균 공급시간 파트 ({bottleneck_name})", bottleneck_supply, "sec"),
    ]
    return "\n".join(
        f"""
        <tr>
          <td>{esc(name)}</td>
          <td class="num">{fmt_num(value)}</td>
          <td>{esc(unit)}</td>
        </tr>
        """
        for name, value, unit in rows
    )


def render_amr_rows(kpi: dict) -> str:
    rows = []
    for record in amr_records(kpi):
        rows.append(
            f"""
            <tr>
              <td>{esc(record["id"])}</td>
              <td class="num">{fmt_percent(record["work_utilization"])}</td>
              <td class="num">{fmt_num(record["work_time"])}</td>
              <td class="num">{fmt_num(record["move_time"])}</td>
              <td class="num">{fmt_num(record["distance"])}</td>
              <td class="num">{fmt_num(record["assignments"])}</td>
              <td class="num">{fmt_num(record["completed_moves"])}</td>
            </tr>
            """
        )
    return "\n".join(rows) or '<tr><td colspan="7" class="empty">AMR KPI가 없습니다.</td></tr>'


def render_depot_rows(kpi: dict) -> str:
    rows = []
    for record in depot_records(kpi):
        depots = ", ".join(record["depots"])
        rows.append(
            f"""
            <tr>
              <td>{esc(record["name"])}</td>
              <td>{esc(str(record["part"]).upper())}</td>
              <td class="num">{fmt_num(record["transfers"])}</td>
              <td class="num">{fmt_num(record["wait_time"])}</td>
              <td class="num">{fmt_num(record["supply_time"])}</td>
              <td class="small">{esc(depots)}</td>
            </tr>
            """
        )
    return "\n".join(rows) or '<tr><td colspan="6" class="empty">파트별 KPI가 없습니다.</td></tr>'


def render_unfinished_depot_rows(kpi: dict) -> str:
    rows = []
    for record in unfinished_depot_records(kpi):
        rows.append(
            f"""
            <tr>
              <td>{esc(record["id"])}</td>
              <td>{esc(record["type"])}</td>
              <td>{esc(record["product"])}</td>
              <td>{esc(record["part"])}</td>
              <td class="num">{fmt_num(record["total"])}</td>
            </tr>
            """
        )
    return "\n".join(rows) or '<tr><td colspan="5" class="empty">병목 KPI가 없습니다.</td></tr>'


def render_bar_chart(caption: str, rows: list[tuple[str, float]], unit: str, max_value: float | None = None) -> str:
    values = [value for _label, value in rows]
    upper = max_value if max_value is not None else (max(values) if values else 0)
    upper = upper if upper > 0 else 1
    body = []
    for label, value in rows:
        width = max(0.0, min((value / upper) * 100, 100.0))
        body.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{esc(label)}</div>
              <div class="bar-track"><div class="bar-fill primary" style="width: {width:.4f}%"></div></div>
              <div class="bar-value">{fmt_num(value)} {esc(unit)}</div>
            </div>
            """
        )
    return f"""
    <figure class="figure">
      <figcaption>{esc(caption)}</figcaption>
      <div class="bar-plot">
        {''.join(body)}
      </div>
    </figure>
    """


def render_grouped_time_chart(kpi: dict) -> str:
    records = depot_records(kpi)
    max_time = max(
        [record["wait_time"] for record in records] + [record["supply_time"] for record in records],
        default=1,
    )
    max_time = max_time if max_time > 0 else 1
    rows = []
    for record in records:
        wait_width = max(0.0, min((record["wait_time"] / max_time) * 100, 100.0))
        supply_width = max(0.0, min((record["supply_time"] / max_time) * 100, 100.0))
        rows.append(
            f"""
            <div class="group-row">
              <div class="bar-label">{esc(record["name"])}</div>
              <div class="group-bars">
                <span class="series-label">대기</span>
                <div class="bar-track"><div class="bar-fill wait" style="width: {wait_width:.4f}%"></div></div>
                <span class="bar-value">{fmt_num(record["wait_time"])} sec</span>
                <span class="series-label">공급</span>
                <div class="bar-track"><div class="bar-fill supply" style="width: {supply_width:.4f}%"></div></div>
                <span class="bar-value">{fmt_num(record["supply_time"])} sec</span>
              </div>
            </div>
            """
        )
    return f"""
    <figure class="figure wide">
      <figcaption>Figure 3. 파트별 평균 대기시간 및 평균 공급시간</figcaption>
      <div class="group-plot">
        {''.join(rows)}
      </div>
    </figure>
    """


def render_html(kpi: dict, source_path: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    amr_chart_rows = [(record["id"], record["work_utilization"]) for record in amr_records(kpi)]
    transfer_chart_rows = [(record["name"], record["transfers"]) for record in depot_records(kpi)]

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>시뮬레이션 KPI 리포트</title>
  <style>
    :root {{
      --page: #f3f4f6;
      --paper: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --rule: #111827;
      --grid: #d1d5db;
      --soft: #f8fafc;
      --blue: #2f5597;
      --gray: #6b7280;
      --dark: #374151;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif;
      font-size: 13px;
      line-height: 1.45;
    }}
    .paper {{
      width: min(1180px, calc(100% - 32px));
      margin: 18px auto 32px;
      background: var(--paper);
      padding: 26px 32px 34px;
      box-shadow: 0 1px 8px rgba(15, 23, 42, 0.12);
    }}
    .report-header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      padding-bottom: 14px;
      border-bottom: 2px solid var(--rule);
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      align-items: flex-start;
      white-space: nowrap;
    }}
    button {{
      height: 32px;
      padding: 0 12px;
      border: 1px solid var(--grid);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
    }}
    .section {{
      margin-top: 18px;
    }}
    .paper-table {{
      width: 100%;
      border-collapse: collapse;
      border-top: 2px solid var(--rule);
      border-bottom: 2px solid var(--rule);
      table-layout: fixed;
      margin-top: 8px;
    }}
    .paper-table caption {{
      caption-side: top;
      text-align: left;
      font-weight: 800;
      font-size: 14px;
      margin-bottom: 6px;
    }}
    .paper-table thead th {{
      background: var(--soft);
      border-bottom: 1.5px solid var(--rule);
      padding: 8px 10px;
      text-align: left;
      font-weight: 800;
    }}
    .paper-table tbody td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      vertical-align: top;
    }}
    .paper-table tbody tr:last-child td {{
      border-bottom: none;
    }}
    .num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .empty {{
      text-align: center;
      color: var(--muted);
      padding: 22px;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    .figure {{
      margin: 0;
      border: 1px solid var(--grid);
      padding: 14px 16px 16px;
      background: #fff;
    }}
    .figure.wide {{
      grid-column: 1 / -1;
    }}
    figcaption {{
      font-weight: 800;
      margin-bottom: 12px;
    }}
    .bar-plot,
    .group-plot {{
      display: grid;
      gap: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 120px minmax(180px, 1fr) 92px;
      gap: 10px;
      align-items: center;
    }}
    .group-row {{
      display: grid;
      grid-template-columns: 130px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
    }}
    .group-bars {{
      display: grid;
      grid-template-columns: 34px minmax(120px, 1fr) 78px 34px minmax(120px, 1fr) 78px;
      gap: 7px;
      align-items: center;
    }}
    .bar-label {{
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .series-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .bar-track {{
      height: 12px;
      background: #edf2f7;
      border: 1px solid #d8dee8;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
    }}
    .bar-fill.primary {{
      background: var(--blue);
    }}
    .bar-fill.wait {{
      background: var(--gray);
    }}
    .bar-fill.supply {{
      background: var(--dark);
    }}
    .bar-value {{
      text-align: right;
      color: var(--ink);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .note {{
      margin: 16px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      .paper {{
        width: calc(100% - 18px);
        padding: 18px 14px 24px;
      }}
      .report-header {{
        flex-direction: column;
      }}
      .figure-grid {{
        grid-template-columns: 1fr;
      }}
      .group-bars {{
        grid-template-columns: 34px minmax(120px, 1fr) 74px;
      }}
      .group-bars .series-label:nth-of-type(2),
      .group-bars .bar-track:nth-of-type(2),
      .group-bars .bar-value:nth-of-type(2) {{
        margin-top: 3px;
      }}
      .section {{
        overflow-x: auto;
      }}
      .paper-table {{
        min-width: 760px;
      }}
    }}
    @media print {{
      body {{
        background: #fff;
      }}
      .paper {{
        width: 100%;
        margin: 0;
        padding: 10mm;
        box-shadow: none;
      }}
      .actions {{
        display: none;
      }}
      .figure {{
        break-inside: avoid;
      }}
      .section {{
        break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>
  <main class="paper">
    <header class="report-header">
      <div>
        <h1>시뮬레이션 KPI 리포트</h1>
        <div class="meta">
          <span>생성시각 {esc(generated_at)}</span>
          <span>source {esc(source_path.name)}</span>
          <span>시뮬레이션 종료시간 {fmt_num(kpi.get("시뮬레이션_종료시간"))} sec</span>
        </div>
      </div>
      <div class="actions">
        <button onclick="window.print()">PDF 내보내기</button>
        <button onclick="window.close()">닫기</button>
      </div>
    </header>

    <section class="section">
      <table class="paper-table">
        <caption>Table 1. 시뮬레이션 총괄 KPI</caption>
        <thead>
          <tr>
            <th>지표</th>
            <th class="num">값</th>
            <th>단위</th>
          </tr>
        </thead>
        <tbody>
          {render_summary_rows(kpi)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="figure-grid">
        {render_bar_chart("Figure 1. 이동로봇별 작업가동률", amr_chart_rows, "%", max_value=100)}
        {render_bar_chart("Figure 2. 파트별 자재데포 이송횟수", transfer_chart_rows, "회")}
        {render_grouped_time_chart(kpi)}
      </div>
    </section>

    <section class="section">
      <table class="paper-table">
        <caption>Table 2. 이동로봇별 작업가동률 및 이송 실적</caption>
        <thead>
          <tr>
            <th>AMR ID</th>
            <th class="num">작업가동률 (%)</th>
            <th class="num">작업가동시간 (sec)</th>
            <th class="num">이동시간 (sec)</th>
            <th class="num">이동거리</th>
            <th class="num">배정횟수</th>
            <th class="num">이동완료횟수</th>
          </tr>
        </thead>
        <tbody>
          {render_amr_rows(kpi)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <table class="paper-table">
        <caption>Table 3. 파트별 자재데포 공급지표</caption>
        <thead>
          <tr>
            <th>자재데포 그룹</th>
            <th>파트</th>
            <th class="num">이송횟수</th>
            <th class="num">평균 대기시간 (sec)</th>
            <th class="num">평균 공급시간 (sec)</th>
            <th>포함 자재데포</th>
          </tr>
        </thead>
        <tbody>
          {render_depot_rows(kpi)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <table class="paper-table">
        <caption>Table 4. 데포별 병목지표</caption>
        <thead>
          <tr>
            <th>데포 ID</th>
            <th>종류</th>
            <th>제품</th>
            <th>파트</th>
            <th class="num">누적</th>
          </tr>
        </thead>
        <tbody>
          {render_unfinished_depot_rows(kpi)}
        </tbody>
      </table>
    </section>

    <p class="note">
      이송횟수는 공정데포 UnLoad 완료 기준이다. 평균 대기시간은 주문 발생부터 ACS 배정까지,
      평균 공급시간은 주문 발생부터 공정데포 UnLoad 완료까지의 시간으로 산정하였다.
      병목은 실제 하차/도착 완료 시점에 해당 데포가 이미 작업 또는 회수 대기 중인 경우만 집계한다.
    </p>
  </main>
</body>
</html>
"""


def write_report(kpi_path: Path | None = None, output_path: Path | None = None) -> Path:
    source_path = kpi_path or find_latest_kpi()
    kpi = load_kpi(source_path)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = LOG_DIR / f"kpi_report_{timestamp}.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(kpi, source_path), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KPI JSON을 논문형 HTML 리포트로 변환합니다.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="입력 KPI JSON 경로. 생략하면 simulation_log의 최신 kpi_*.json을 사용합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 HTML 경로. 생략하면 simulation_log/kpi_report_*.html로 저장합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_report(args.input, args.output)
    print(f"KPI HTML 리포트 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
