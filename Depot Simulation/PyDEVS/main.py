import argparse

from runtime_product import add_product_argument, choose_product, configure_product_env


def find_model(engine, model_id):
    for model in engine.models:
        if getattr(model, "getModelID", lambda: None)() == model_id:
            return model
    return None


def print_kpi_summary(kpi):
    print("\n========== KPI 요약 ==========")
    print(f"시뮬레이션 종료시간: {kpi.get('시뮬레이션_종료시간')}")
    print(f"총 작업배정횟수: {kpi.get('총_작업배정횟수')}")
    print(f"총 자재데포 -> 공정데포 이송횟수: {kpi.get('총_자재데포_공정데포_이송횟수')}")
    print(f"총 공정데포 병목횟수: {kpi.get('총_공정데포_병목횟수')}")
    print(f"총 회수데포 병목횟수: {kpi.get('총_회수데포_병목횟수')}")
    print(f"평균 대기시간: {kpi.get('평균_대기시간')}")
    print(f"평균 공급시간: {kpi.get('평균_공급시간')}")

    robot_utilization = kpi.get('이동로봇별_가동률') or {}
    if robot_utilization:
        print("\n이동로봇별 가동률")
        for amr_id, values in robot_utilization.items():
            percent = values.get('가동률_퍼센트')
            move_time = values.get('이동시간')
            distance = values.get('이동거리')
            assigned = values.get('배정횟수')
            print(
                f"- {amr_id}: {percent}% "
                f"(이동시간={move_time}, 이동거리={distance}, 배정횟수={assigned})"
            )

    material_depots = kpi.get('파트별_자재데포_공급지표') or {}
    if material_depots:
        print("\n파트별 자재데포 공급지표")
        for group_name, values in material_depots.items():
            print(
                f"- {group_name}: 이송횟수={values.get('이송횟수')}, "
                f"평균 대기시간={values.get('평균_대기시간')}, "
                f"평균 공급시간={values.get('평균_공급시간')}"
            )

    bottlenecks = kpi.get('데포별_병목지표') or {}
    if bottlenecks:
        print("\n데포별 병목지표")
        for depot_id, values in bottlenecks.items():
            print(
                f"- {depot_id}: 누적={values.get('누적_병목횟수')}, "
                f"종류={values.get('데포종류')}"
            )

    print("==============================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PyDEVS depot simulation.")
    add_product_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_product = choose_product(args.product)
    configure_product_env(selected_product)

    from SimulationEngine.SimulationEngine import SimulationEngine
    from SimulationEngine.Utility.Configurator import Configurator
    from modeling.outmost import Outmost
    from config import AMR_LIST, PRODUCT_LABEL, SELECTED_PRODUCT, SIMULATION_MAX_TIME

    print(f"실행 제품: {PRODUCT_LABEL} ({SELECTED_PRODUCT})")

    objConfiguration = Configurator()
    objConfiguration.addConfiguration("AMR_LIST", list(AMR_LIST))

    outmost = Outmost(objConfiguration)

    engine = SimulationEngine()
    engine.setOutmostModel(outmost)
    engine.run(
        maxTime=SIMULATION_MAX_TIME,
        ta=-1,
        visualizer=False,
        logFileName=-1,
        logGeneral=False,
        logActivateState=False,
        logActivateMessage=False,
        logActivateTA=False,
        logStructure=False,
    )

    analyzer = find_model(engine, "Analyzer")
    if analyzer is not None and hasattr(analyzer, "get_kpi"):
        if hasattr(analyzer, "export_kpi"):
            analyzer.export_kpi()
        print_kpi_summary(analyzer.get_kpi())


if __name__ == "__main__":
    main()
