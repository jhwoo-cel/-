from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
from collections import deque
import heapq
import math
import numpy as np
import json
from pathlib import Path
from config import AMR_CELL_TRAVEL_TIME, AMR_LOAD_UNLOAD_TIME, MAP_PATH

# CHECK랑 TRANSMIT 다시보기

INF = float('inf')
BLOCKED_PATH_EDGES = {
    frozenset(("N170", "N281")),
    frozenset(("N271", "N288")),
}
WH_DEPOT_AREA_MARGIN_CELLS = 4
DEPOT_AREA_EXCLUDED_PATH_NODE_NAMES = {"N4687", "N4768"}


class ACS(DEVSAtomicModel):

    def __init__(self, ID, map_path=None, objConfiguration=None):
        super().__init__(ID)
        self.objConfiguration = objConfiguration

        if map_path is None:
            map_path = MAP_PATH


        self.map_data = self.load_map(map_path)
        self.location_index = self.build_location_index(self.map_data)
        self.grid_size = self.get_grid_size(self.map_data)
        self.path_node_by_point = self.build_path_node_index(self.map_data)
        self.path_points = set(self.path_node_by_point)
        self.depot_group_by_point = self.build_depot_group_index(self.map_data)
        self.depot_area_by_point = self.build_depot_area_index(self.map_data)
        self.obstacle_points = self.build_obstacle_points(self.map_data)
        self.blocked_points = self.build_blocked_points(self.map_data)
        self.home_station_by_amr = self.build_home_station_index()
        self.cell_travel_ticks = max(1, int(math.ceil(AMR_CELL_TRAVEL_TIME)))
        self.load_unload_ticks = max(1, int(math.ceil(AMR_LOAD_UNLOAD_TIME)))
        
        self.addStateVariable('state', 'IDLE')

        self.addInputPort('schedule_info_from_MCS')
        self.addInputPort('amr_info_from_tm')

        self.addOutputPort('amr_req_to_tm')
        self.addOutputPort('amr_update_to_tm')
        self.addOutputPort('cmd_to_amr')
        self.addOutputPort('operation_result_to_analyzer')


        self.schedule_queue = []
        self.current_task = None

        self.schedule_msg = None
        self.amr_msg = None
        self.cmd_msg = None
        self.selected_amr = None
        self.assignment_sequence = 0
        self.last_assignment_sequence_by_amr = {}
        self.command_failed = False

        self.amr_request_sent = False
        self.move_num = 0
        


    def funcExternalTransition(self, strPort, event): # 외부상태 천이
        state = self.getStateValue('state')
        if strPort == 'schedule_info_from_MCS':
            self.schedule_msg = event # mcs에서 오는 스케줄 정보
            self.schedule_queue.append(self.schedule_msg) # 스케줄 정보를 큐에 등록
            if state == 'IDLE' or (state == 'WAIT' and self.current_task is None):
                self.setStateValue('state', 'CHECK')
            else:
                self.setStateValue('state', state)
            return

        if state == 'IDLE': # 작성
            if strPort == 'amr_info_from_tm': # 작성. tm에서 업데이트 되면 자동으로 보내는 것
                self.amr_msg=event # tm으로부터 빈 amr 정보가 올 때도 있을 것
                if self.current_task is not None or len(self.schedule_queue) > 0: # 작업 가능한 스케줄이 있으면 check로 보냄
                    self.setStateValue('state', 'CHECK')
                else: # 없으면 idle에서 스케줄 들어올때까지 대기
                    self.setStateValue('state', 'IDLE')
        elif state == 'CHECK':
            pass
        elif state == 'WAIT': # 작성
            self.amr_msg= event # tm에서 오는 idle amr 정보. 얘도 빈 리스트일 수도 있음
            if strPort == 'amr_info_from_tm':
                self.setStateValue('state','CHECK')
        elif state == 'COMMAND':
            pass
        elif state == 'TRANSMIT':
            pass

    def funcOutput(self): # 출력
        state = self.getStateValue('state')
        if state == 'IDLE':
            pass
        elif state == 'CHECK': # 요청은 schedule info from mcs에서 온 정보 리스트 중 가장 첫번째에 있는 거
            # 현재 처리 중인 작업이 없고, 대기 중인 스케줄 큐에 작업이 있으면
            # 큐의 가장 앞 작업을 꺼내 이번에 처리할 current_task로 등록
            if self.current_task is None and len(self.schedule_queue) > 0:
                self.amr_msg = None
                self.amr_request_sent = False
                self.current_task = self.schedule_queue.pop(0) # 현재 작업 중인 거 없고 스케줄 받은 거 있으면 현재 작업에 그 스케줄 등록
            
            # 현재 처리할 작업은 있지만 아직 사용할 AMR 후보 정보가 없으며
            # 이 작업에 대해 amr 후보 요청도 아직 보내지 않았다면
            if (
                self.current_task is not None
                and self.amr_msg is None
                and not self.amr_request_sent
            ):
                # tm에게 현재 작업을 수행할 수 있는 amr 후보 요청. current_task에 amr_status를 포함해서 보내야함. 그래야지 amr에서 상태 변경 할 수 있음. undispatchable로 
                self.addOutputEvent('amr_req_to_tm', self.current_task)

                # 같은 작업에 대해 중복으로 요청을 보내지 않도록 요청 완료 표시
                self.amr_request_sent = True
                print(f"ACS에서 요청{self.current_task}를 tm으로 보냄")

        elif state == 'WAIT':
            pass
        elif state == 'COMMAND':
            task = self.current_task

            from_info = self.get_location_info(task.from_)
            to_info = self.get_location_info(task.to)
            from_id = self.get_location_id(from_info)
            to_id = self.get_location_id(to_info)
            part = task.part
            task_type = task.task_type or "move"

            from_coord = self.get_location_coord(from_info)
            to_coord = self.get_location_coord(to_info)
            recovery_from_info = self.get_location_info(task.recovery_from)
            recovery_to_info = self.get_location_info(task.recovery_to)
            recovery_from_coord = (
                self.get_location_coord(recovery_from_info)
                if recovery_from_info is not None
                else None
            )
            recovery_to_coord = (
                self.get_location_coord(recovery_to_info)
                if recovery_to_info is not None
                else None
            )

            amr_list = self.amr_msg.amr_list or []
            selected_amr = self.select_amr(amr_list, from_coord, task.product)

            if selected_amr is None:
                print("ACS: 배정 가능한 AMR이 없습니다.")
                self.command_failed = True
                self.amr_msg = None
                self.amr_request_sent = False
                return

            self.selected_amr = selected_amr
            self.assignment_sequence += 1
            self.last_assignment_sequence_by_amr[selected_amr.AMR_id] = self.assignment_sequence
            self.move_num += 1

            amr_coord = (
                selected_amr.x,
                selected_amr.y
            )

            current_coord = {
                "x": selected_amr.x,
                "y": selected_amr.y
            }

            station_id = self.select_station(selected_amr)
            station_coord = self.get_coord(station_id)

            route = self.make_full_route(
                amr_coord=amr_coord,
                from_coord=from_coord,
                to_coord=to_coord,
                station_coord=station_coord,
                recovery_from_coord=recovery_from_coord,
                recovery_to_coord=recovery_to_coord,
                initial_leg=task.initial_leg,
                amr_id=selected_amr.AMR_id,
                start_time=self.getTime()
            )

            self.cmd_msg = Message(
                AMR_id=selected_amr.AMR_id,
                task_id=task.task_id or f"move_{self.move_num:06d}",
                task_type=task_type,
                task_step="dispatch",

                product=task.product,
                from_=from_info,
                to=to_info,
                part=part,
                cart_count=task.cart_count,
                generated_time=task.generated_time,
                target_slot=task.target_slot,
                target_label=task.target_label,
                chain_recovery=task.chain_recovery,
                recovery_from=recovery_from_info,
                recovery_to=recovery_to_info,
                recovery_ready_time=task.recovery_ready_time,
                initial_leg=task.initial_leg,

                current_location=selected_amr.current_location,
                goal_type="station",
                goal_location=station_id,

                current_coord=current_coord,
                from_coord={
                    "x": from_coord[0],
                    "y": from_coord[1]
                },
                to_coord={
                    "x": to_coord[0],
                    "y": to_coord[1]
                },
                station_coord={
                    "x": station_coord[0],
                    "y": station_coord[1]
                },

                route=route,
                move_num=self.move_num,
                battery_level=selected_amr.battery_level,
                amr_status="DISPATCHED",
                event_description="ACS dispatches AMR and sends route command"
            )

            self.addOutputEvent("cmd_to_amr", self.cmd_msg)

            print(
                f"ACS: {selected_amr.AMR_id}를 "
                f"{from_id} -> {to_id} 작업에 배정함"
            )


        elif state == 'TRANSMIT': # 작성 # 할당된 amr은 이 부분에서 빼줘야 해야할 듯
                amr_id = self.selected_amr.AMR_id
                move_num = self.cmd_msg.move_num

                amr_update_msg = Message(
                    AMR_id=amr_id,
                    amr_status="DISPATCHED",
                    idle_type=None,
                    event_description="ACS updates assigned AMR to TM"
                )

                operation_result_msg = Message(
                    AMR_id=amr_id,
                    task_id=self.cmd_msg.task_id,
                    product=self.cmd_msg.product,
                    part=self.cmd_msg.part,
                    cart_count=self.cmd_msg.cart_count,
                    generated_time=self.cmd_msg.generated_time,
                    from_=self.cmd_msg.from_,
                    to=self.cmd_msg.to,
                    move_num=move_num,
                    timestamp=self.getTime(),
                    result_status="ASSIGNED",
                    event_description="ACS sends operation result to analyzer"
                )

                self.addOutputEvent("amr_update_to_tm", amr_update_msg)
                self.addOutputEvent("operation_result_to_analyzer", operation_result_msg)

                print( 
                    f"ACS: TM에 AMR 업데이트 전송 - "
                    f"AMR_id={amr_update_msg.AMR_id}, "
                    f"amr_status={amr_update_msg.amr_status}, "
                    f"idle_type={amr_update_msg.idle_type}")
                print(
                    f"ACS: Analyzer에 operation result 전송 - "
                    f"AMR_id={operation_result_msg.AMR_id}, "
                    f"move_num={operation_result_msg.move_num}, "
                    f"result_status={operation_result_msg.result_status}"
                )

    def funcInternalTransition(self): # 내부상태 천이
        state = self.getStateValue('state')
        if state == 'IDLE':
            pass
        elif state == 'CHECK': # 다시 보기
            if self.amr_msg is None: # tm으로 받은 amr 정보가 없어 wait로 정보 요청
                print(
                    f"ACS CHECK: AMR 후보 정보가 아직 없음. "
                    f"current_task={self.current_task} → WAIT로 전달"
                )
                self.setStateValue('state','WAIT')
            elif len(self.amr_msg.amr_list or []) == 0: # 응답은 받았는데 매칭 가능한 AMR이 없음 → IDLE 복귀
                self.amr_msg = None # 빈 응답 폐기
                self.amr_request_sent = False # 다시 요청 가능하게 request 플래그 초기화
                # TM 응답을 받았는데 처음부터 후보 AMR이 하나도 없었던 경우를 처리
                self.setStateValue('state','IDLE')
            else:
                self.setStateValue('state','COMMAND')
        elif state == 'WAIT':
            pass
        elif state == 'COMMAND':
            if self.command_failed:
                self.command_failed = False
                self.setStateValue('state', 'IDLE')
            else:
                self.setStateValue('state','TRANSMIT')
        elif state == 'TRANSMIT':
            # 1. 이번에 배정한 AMR만 ACS가 들고 있는 후보 리스트에서 제거
            if self.selected_amr is not None and self.amr_msg is not None:
                selected_amr_id = self.selected_amr.AMR_id  # 선택된 AMR

                # self.amr_msg.amr_list 안에 있는 AMR 후보들 중에서
                # 이번에 선택된 AMR_id와 다른 AMR만 다시 리스트로 만든다.
                self.amr_msg.amr_list = [
                    amr for amr in (self.amr_msg.amr_list or [])
                    if amr.AMR_id != selected_amr_id
                ]

                # 남은 AMR 후보가 없으면 다음 작업 때 TM에 새로 요청해야 함
                if len(self.amr_msg.amr_list or []) == 0:
                    self.amr_msg = None
                    self.amr_request_sent = False  # 다시 TM에 AMR 후보 요청 가능

                    print(
                        f"ACS TRANSMIT: 배정 후 남은 AMR 후보 없음. "
                        f"다음 작업 때 TM에 새로 요청"
                    )

                else:
                    # 남은 AMR 후보가 있으면 그대로 유지
                    self.amr_request_sent = True

                    print(
                        f"ACS TRANSMIT: 배정 가능한 AMR 후보 수="
                        f"{len(self.amr_msg.amr_list or [])}. "
                        f"남은 후보 리스트 유지"
                    )

            # 2. 이번 작업에 사용한 임시 정보 삭제
            self.current_task = None      # schedule_queue에서 pop(0)으로 꺼낸 이번 작업 하나
            self.selected_amr = None      # 후보 중 이번 작업에 배정한 AMR 하나
            self.cmd_msg = None           # 이번 작업에 대해 AMR에게 보낸 command 메시지 하나

            # 3. 대기 스케줄이 있으면 바로 다음 작업을 확인한다.
            if len(self.schedule_queue) > 0:
                self.setStateValue('state', 'CHECK')
            else:
                self.setStateValue('state', 'IDLE')



    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'IDLE':
            return INF
        elif state == 'CHECK':
            return 0
        elif state == 'WAIT':
            return INF
        elif state == 'COMMAND':
            return 0
        elif state == 'TRANSMIT':
            return 0

    def funcSelect(self):
        pass

    def load_map(self, map_path):
        map_path = Path(map_path)

        if not map_path.exists():
            raise FileNotFoundError(f"map.json 파일을 찾을 수 없습니다: {map_path}")

        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def build_location_index(self, map_data):
        location_index = {}

        for group_name in ["wh_locations", "ml_nodes", "ps_nodes"]:
            for node in map_data.get(group_name, []):
                location_id = node["location_id"]
                location_index[location_id] = node

        return location_index

    def build_path_node_index(self, map_data):
        path_node_by_point = {}
        for node in map_data.get("amr_path_nodes", []):
            if node.get("x") is None or node.get("y") is None:
                continue
            path_node_by_point[(int(node["x"]), int(node["y"]))] = node.get("name")
        return path_node_by_point

    def build_depot_group_index(self, map_data):
        depot_group_by_point = {}
        for node in map_data.get("wh_locations", []):
            coord = node.get("coordinates") or {}
            if coord.get("x") is None or coord.get("y") is None:
                continue
            depot_group_by_point[(int(coord["x"]), int(coord["y"]))] = (
                "WH",
                node.get("product"),
                node.get("section"),
                node.get("part"),
            )

        for node in map_data.get("ml_nodes", []):
            coord = node.get("coordinates") or {}
            if coord.get("x") is None or coord.get("y") is None:
                continue
            depot_group_by_point[(int(coord["x"]), int(coord["y"]))] = (
                "ML",
                node.get("product"),
                node.get("part"),
            )

        for node in map_data.get("ps_nodes", []):
            coord = node.get("coordinates") or {}
            if coord.get("x") is None or coord.get("y") is None:
                continue
            depot_group_by_point[(int(coord["x"]), int(coord["y"]))] = (
                "PS",
                node.get("product"),
                node.get("amr_type"),
            )

        return depot_group_by_point

    def depot_area_key(self, group_name, node):
        if group_name == "wh_locations":
            return (
                "WH",
                node.get("product"),
                node.get("section"),
                node.get("part"),
            )
        if group_name == "ml_nodes":
            return (
                "ML",
                node.get("product"),
                node.get("part"),
                node.get("rack_type"),
            )
        if group_name == "ps_nodes":
            return (
                "PS",
                node.get("product"),
                node.get("amr_type"),
            )
        return None

    def build_depot_area_index(self, map_data):
        grouped_points = {}

        for group_name in ("wh_locations", "ml_nodes", "ps_nodes"):
            for node in map_data.get(group_name, []):
                coord = node.get("coordinates") or {}
                if coord.get("x") is None or coord.get("y") is None:
                    continue

                key = self.depot_area_key(group_name, node)
                if key is None:
                    continue

                grouped_points.setdefault(key, []).append(
                    (int(coord["x"]), int(coord["y"]))
                )

        area_by_point = {}
        for key, points in grouped_points.items():
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)

            margin = WH_DEPOT_AREA_MARGIN_CELLS if key[0] == "WH" else 0
            min_x = max(0, min_x - margin)
            min_y = max(0, min_y - margin)
            max_x = min(self.grid_size[0] - 1, max_x + margin)
            max_y = min(self.grid_size[1] - 1, max_y + margin)

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    point = (x, y)
                    if self.path_node_by_point.get(point) in DEPOT_AREA_EXCLUDED_PATH_NODE_NAMES:
                        continue
                    area_by_point.setdefault(point, set()).add(key)

        return area_by_point

    def build_obstacle_points(self, map_data):
        obstacle_points = set()

        for key, cell in (map_data.get("cell_map") or {}).items():
            if cell.get("type") != "obstacle":
                continue
            try:
                x_text, y_text = key.split(",", 1)
                obstacle_points.add((int(x_text), int(y_text)))
            except (TypeError, ValueError):
                continue

        for region in map_data.get("regions", []):
            if region.get("type") != "obstacle":
                continue
            for cell in region.get("cells", []):
                if len(cell) < 2:
                    continue
                obstacle_points.add((int(cell[0]), int(cell[1])))

        return obstacle_points

    def get_grid_size(self, map_data):
        grid = map_data.get("metadata", {}).get("grid", {})
        width = grid.get("width")
        height = grid.get("height")

        if width is not None and height is not None:
            return int(width), int(height)

        max_x = 0
        max_y = 0
        for node in self.location_index.values():
            coord = node.get("coordinates") or {}
            if coord.get("x") is not None:
                max_x = max(max_x, int(coord["x"]))
            if coord.get("y") is not None:
                max_y = max(max_y, int(coord["y"]))

        return max_x + 1, max_y + 1

    def build_blocked_points(self, map_data):
        blocked_points = set()

        for group_name in ["wh_locations", "ml_nodes", "ps_nodes"]:
            for node in map_data.get(group_name, []):
                coord = node.get("coordinates") or {}
                if coord.get("x") is not None and coord.get("y") is not None:
                    blocked_points.add((int(coord["x"]), int(coord["y"])))

        return blocked_points

    def get_location_info(self, location):
        if isinstance(location, list):
            return location[0] if location else None
        return location

    def get_location_id(self, location):
        if isinstance(location, dict):
            return location.get("location_id")
        return location

    def get_location_coord(self, location):
        if isinstance(location, dict):
            coord = location.get("coordinates")
            if coord is not None:
                return coord["x"], coord["y"]
        return self.get_coord(self.get_location_id(location))
    
    def get_coord(self, location_id):
        node = self.location_index.get(location_id)

        if node is None:
            raise KeyError(f"map.json에 없는 location_id입니다: {location_id}")

        coord = node["coordinates"]
        return coord["x"], coord["y"]
    
    def manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def select_amr(self, amr_list, target_coord, product=None):
        best_amr = None
        best_priority = None

        for amr in amr_list:
            amr_product = getattr(amr, "amr_product", None)
            if product is not None and amr_product is not None and amr_product != product:
                continue

            amr_id = amr.AMR_id
            last_assignment = self.last_assignment_sequence_by_amr.get(amr_id, -1)
            amr_coord = (amr.x, amr.y)
            distance = self.manhattan_distance(amr_coord, target_coord)
            idle_type = getattr(amr, "idle_type", None)
            status = getattr(amr, "status", None)
            rest_priority = 0 if idle_type == "STATION_IDLE" or status == "parking" else 1
            priority = (rest_priority, distance, last_assignment, amr_id)

            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_amr = amr

        return best_amr
    
    def build_home_station_index(self):
        home_station_by_amr = {}
        if self.objConfiguration is None:
            return home_station_by_amr

        for config in self.objConfiguration.getConfiguration("AMR_LIST") or []:
            amr_id = config.get("AMR_id")
            current_location = config.get("current_location")
            if amr_id is not None and current_location is not None:
                home_station_by_amr[amr_id] = current_location
        return home_station_by_amr

    def select_station(self, selected_amr):
        home_station = self.home_station_by_amr.get(selected_amr.AMR_id)
        if home_station is not None:
            return home_station

        amr_type = selected_amr.amr_type
        for node in self.map_data.get("ps_nodes", []):
            if node.get("amr_type") == amr_type and node.get("occupied") == 0:
                return node["location_id"]

        for node in self.map_data.get("ps_nodes", []):
            if node.get("amr_type") == amr_type:
                return node["location_id"]

        ps_nodes = self.map_data.get("ps_nodes", [])
        if len(ps_nodes) > 0:
            return ps_nodes[0]["location_id"]

        raise ValueError("사용 가능한 station이 없습니다.")
    
    def make_manhattan_route(self, start, goal):
        x1, y1 = start
        x2, y2 = goal

        route = []
        x, y = x1, y1

        while x != x2:
            x += 1 if x2 > x else -1
            route.append({"x": x, "y": y})

        while y != y2:
            y += 1 if y2 > y else -1
            route.append({"x": x, "y": y})

        return route

    def as_point(self, point):
        if isinstance(point, dict):
            return int(point["x"]), int(point["y"])
        return int(point[0]), int(point[1])

    def is_in_grid(self, point):
        x, y = point
        width, height = self.grid_size
        return 0 <= x < width and 0 <= y < height

    def raw_neighbors(self, point):
        x, y = point
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

    def is_blocked_path_edge(self, point_a, point_b):
        name_a = self.path_node_by_point.get(point_a)
        name_b = self.path_node_by_point.get(point_b)
        if name_a is None or name_b is None:
            return False
        return frozenset((name_a, name_b)) in BLOCKED_PATH_EDGES

    def is_connected_path_point(self, point):
        if point not in self.path_points:
            return False
        for neighbor in self.raw_neighbors(point):
            if neighbor in self.path_points and not self.is_blocked_path_edge(point, neighbor):
                return True
        return False

    def is_access_passable(self, candidate, allowed_depot_area_keys=None):
        if not self.is_in_grid(candidate):
            return False
        if candidate not in self.path_points:
            return False

        allowed_depot_area_keys = set(allowed_depot_area_keys or set())
        depot_area_keys = self.depot_area_by_point.get(candidate, set())
        if depot_area_keys and not depot_area_keys.intersection(allowed_depot_area_keys):
            return False

        return True

    def build_access_path_to_amr_path(self, point, allowed_depot_area_keys=None):
        point = self.as_point(point)
        allowed_depot_area_keys = set(allowed_depot_area_keys or set())
        if not self.path_points:
            return {point}
        if point not in self.path_points:
            return set()
        if self.is_connected_path_point(point) and not self.depot_area_by_point.get(point):
            return {point}

        queue = deque([point])
        came_from = {point: None}
        target = None

        while queue:
            current = queue.popleft()
            if (
                current != point
                and current in self.path_points
                and not self.depot_area_by_point.get(current)
            ):
                target = current
                break

            for candidate in self.raw_neighbors(current):
                if candidate in came_from:
                    continue
                if candidate not in self.path_points:
                    continue
                if not self.is_access_passable(candidate, allowed_depot_area_keys):
                    continue
                came_from[candidate] = current
                queue.append(candidate)

        if target is None:
            return {point}

        access_points = set()
        current = target
        while current is not None:
            access_points.add(current)
            current = came_from[current]
        return access_points

    def build_allowed_route_points(self, allowed_points):
        if not self.path_points:
            return None

        allowed_points = [self.as_point(point) for point in (allowed_points or [])]
        allowed_depot_area_keys = set()
        for point in allowed_points:
            allowed_depot_area_keys.update(self.depot_area_by_point.get(point, set()))

        allowed = {
            point
            for point in self.path_points
            if not self.depot_area_by_point.get(point)
        }
        for route_point in allowed_points:
            allowed.update(
                self.build_access_path_to_amr_path(
                    route_point,
                    allowed_depot_area_keys,
                )
            )
            if route_point in self.path_points:
                allowed.add(route_point)
        return allowed

    def get_route_neighbors(self, point, blocked_points, allowed_route_points=None):
        for candidate in self.raw_neighbors(point):
            if not self.is_in_grid(candidate):
                continue
            if allowed_route_points is not None and candidate not in allowed_route_points:
                continue
            if self.is_blocked_path_edge(point, candidate):
                continue
            if candidate not in blocked_points:
                yield candidate

    def time_to_tick(self, time_value):
        return int(round(float(time_value or 0.0)))

    def step_duration_ticks(self, previous_point, next_point):
        return 1 if previous_point == next_point else self.cell_travel_ticks

    def route_duration_ticks(self, start, path):
        previous_point = self.as_point(start)
        duration = 0
        for point in path:
            next_point = self.as_point(point)
            duration += self.step_duration_ticks(previous_point, next_point)
            previous_point = next_point
        return duration

    def reconstruct_route(self, came_from, start, goal):
        current = goal
        path = []

        while current != start:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return [{"x": x, "y": y} for x, y in path]

    def reconstruct_timed_route(self, came_from, start_state, goal_state):
        current = goal_state
        path = []

        while current != start_state:
            point, _ = current
            path.append(point)
            current = came_from[current]

        path.reverse()
        return [{"x": x, "y": y} for x, y in path]

    def make_static_astar_route(self, start, goal, allowed_points=None):
        start = self.as_point(start)
        goal = self.as_point(goal)

        if start == goal:
            return []

        allowed = {start, goal}
        for point in allowed_points or []:
            allowed.add(self.as_point(point))

        allowed_route_points = self.build_allowed_route_points(allowed)
        if allowed_route_points is None:
            blocked_points = self.blocked_points - allowed
        else:
            blocked_points = self.blocked_points - allowed_route_points

        open_heap = []
        start_priority = self.manhattan_distance(start, goal)
        heapq.heappush(open_heap, (start_priority, 0, start))

        came_from = {}
        cost_so_far = {start: 0}
        visited = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                return self.reconstruct_route(came_from, start, goal)

            for next_point in self.get_route_neighbors(current, blocked_points, allowed_route_points):
                next_cost = current_cost + 1

                if next_cost >= cost_so_far.get(next_point, float("inf")):
                    continue

                cost_so_far[next_point] = next_cost
                came_from[next_point] = current

                priority = next_cost + self.manhattan_distance(next_point, goal)
                heapq.heappush(open_heap, (priority, next_cost, next_point))

        raise RuntimeError(
            "ACS: AMR path-node route not found. "
            f"start={start}, goal={goal}"
        )
    
    def make_astar_route(self, start, goal, allowed_points=None, start_time=None, amr_id=None):
        return self.make_static_astar_route(start, goal, allowed_points)
    
    def make_full_route(
        self,
        amr_coord,
        from_coord,
        to_coord,
        station_coord,
        recovery_from_coord=None,
        recovery_to_coord=None,
        initial_leg=None,
        amr_id=None,
        start_time=None,
    ):
        start_tick = self.time_to_tick(start_time if start_time is not None else self.getTime())

        if initial_leg == "current_to_recovery":
            current_to_recovery = self.make_astar_route(
                amr_coord,
                from_coord,
                allowed_points=[amr_coord, from_coord],
                start_time=start_tick,
                amr_id=amr_id
            )
            current_to_recovery_ticks = self.route_duration_ticks(amr_coord, current_to_recovery)
            recovery_to_empty_start_tick = start_tick + current_to_recovery_ticks + self.load_unload_ticks

            recovery_to_empty = self.make_astar_route(
                from_coord,
                to_coord,
                allowed_points=[from_coord, to_coord],
                start_time=recovery_to_empty_start_tick,
                amr_id=amr_id
            )
            recovery_to_empty_ticks = self.route_duration_ticks(from_coord, recovery_to_empty)
            empty_to_station_start_tick = recovery_to_empty_start_tick + recovery_to_empty_ticks + self.load_unload_ticks

            empty_to_station = self.make_astar_route(
                to_coord,
                station_coord,
                allowed_points=[to_coord, station_coord],
                start_time=empty_to_station_start_tick,
                amr_id=amr_id
            )

            load_wait = [{"x": from_coord[0], "y": from_coord[1]} for _ in range(self.load_unload_ticks)]
            unload_wait = [{"x": to_coord[0], "y": to_coord[1]} for _ in range(self.load_unload_ticks)]
            return {
                "current_to_recovery": current_to_recovery,
                "recovery_to_empty": recovery_to_empty,
                "empty_to_station": empty_to_station,
            }

        current_to_from = self.make_astar_route(
            amr_coord,
            from_coord,
            allowed_points=[amr_coord, from_coord],
            start_time=start_tick,
            amr_id=amr_id
        )
        current_to_from_ticks = self.route_duration_ticks(amr_coord, current_to_from)
        from_to_start_tick = start_tick + current_to_from_ticks + self.load_unload_ticks

        from_to = self.make_astar_route(
            from_coord,
            to_coord,
            allowed_points=[from_coord, to_coord],
            start_time=from_to_start_tick,
            amr_id=amr_id
        )
        from_to_ticks = self.route_duration_ticks(from_coord, from_to)
        next_start_tick = from_to_start_tick + from_to_ticks + self.load_unload_ticks

        load_wait = [{"x": from_coord[0], "y": from_coord[1]} for _ in range(self.load_unload_ticks)]
        unload_wait = [{"x": to_coord[0], "y": to_coord[1]} for _ in range(self.load_unload_ticks)]

        if recovery_from_coord is not None and recovery_to_coord is not None:
            to_recovery = self.make_astar_route(
                to_coord,
                recovery_from_coord,
                allowed_points=[to_coord, recovery_from_coord],
                start_time=next_start_tick,
                amr_id=amr_id
            )
            to_recovery_ticks = self.route_duration_ticks(to_coord, to_recovery)
            recovery_to_empty_start_tick = next_start_tick + to_recovery_ticks + self.load_unload_ticks

            recovery_to_empty = self.make_astar_route(
                recovery_from_coord,
                recovery_to_coord,
                allowed_points=[recovery_from_coord, recovery_to_coord],
                start_time=recovery_to_empty_start_tick,
                amr_id=amr_id
            )
            recovery_to_empty_ticks = self.route_duration_ticks(recovery_from_coord, recovery_to_empty)
            empty_to_station_start_tick = (
                recovery_to_empty_start_tick
                + recovery_to_empty_ticks
                + self.load_unload_ticks
            )

            empty_to_station = self.make_astar_route(
                recovery_to_coord,
                station_coord,
                allowed_points=[recovery_to_coord, station_coord],
                start_time=empty_to_station_start_tick,
                amr_id=amr_id
            )

            recovery_load_wait = [
                {"x": recovery_from_coord[0], "y": recovery_from_coord[1]}
                for _ in range(self.load_unload_ticks)
            ]
            empty_unload_wait = [
                {"x": recovery_to_coord[0], "y": recovery_to_coord[1]}
                for _ in range(self.load_unload_ticks)
            ]

            return {
                "current_to_from": current_to_from,
                "from_to": from_to,
                "to_recovery": to_recovery,
                "recovery_to_empty": recovery_to_empty,
                "empty_to_station": empty_to_station,
            }

        to_station = self.make_astar_route(
            to_coord,
            station_coord,
            allowed_points=[to_coord, station_coord],
            start_time=next_start_tick,
            amr_id=amr_id
        )

        return {
            "current_to_from": current_to_from,
            "from_to": from_to,
            "to_station": to_station,
        }
