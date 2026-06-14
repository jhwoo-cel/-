from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
import json
from pathlib import Path
from config import (
    AMR_ACCELERATION_M_PER_SEC2,
    AMR_CELL_TRAVEL_TIME,
    AMR_DECELERATION_M_PER_SEC2,
    AMR_DOCKING_DISTANCE_M,
    AMR_DOCKING_SPEED_M_PER_SEC,
    AMR_EMPTY_CART_MAX_SPEED_M_PER_SEC,
    AMR_LOADED_MAX_SPEED_M_PER_SEC,
    AMR_PHYSICAL_LOG_INTERVAL_SEC,
    AMR_LOAD_UNLOAD_TIME,
    AMR_ROUTE_POINT_MODE,
    AMR_SPEED_M_PER_SEC,
    AMR_TURN_TIME_SEC,
    AMR_UNLOADED_MAX_SPEED_M_PER_SEC,
    MAP_PATH,
    MAP_CELL_SIZE_M,
)


INF = float('inf')


def _load_core_path_points():
    core_path = Path(MAP_PATH).resolve().parent / "core_amr_node.json"
    if not core_path.exists():
        return set()

    try:
        with core_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return set()

    points = set()
    for node in data.get("core_amr_path_nodes", []):
        if node.get("x") is None or node.get("y") is None:
            continue
        points.add((int(node["x"]), int(node["y"])))
    return points


CORE_AMR_PATH_POINTS = _load_core_path_points()


class AMR(DEVSAtomicModel):

    def __init__(
        self,
        ID,
        speed=AMR_SPEED_M_PER_SEC,
        load_unload_time=AMR_LOAD_UNLOAD_TIME,
        loaded_max_speed=None,
        unloaded_max_speed=None,
        empty_cart_max_speed=None,
        acceleration=None,
        deceleration=None,
        docking_speed=None,
        docking_distance=None,
        turn_time=None,
    ):
        super().__init__(ID)

        self.addStateVariable('state', 'IDLE')

        self.addInputPort('cmd_from_ACS')

        self.addOutputPort('cmd_completed_to_mm')
        self.addOutputPort('return_completed_to_tm')
        self.addOutputPort('physical_result_to_analyzer')
        self.addOutputPort('physical_result_to_tm')


        self.AMR_id = ID
        self.nominal_speed = float(speed)
        self.speed = float(speed)
        self.loaded_max_speed = float(loaded_max_speed or AMR_LOADED_MAX_SPEED_M_PER_SEC)
        self.unloaded_max_speed = float(unloaded_max_speed or AMR_UNLOADED_MAX_SPEED_M_PER_SEC)
        self.empty_cart_max_speed = float(empty_cart_max_speed or AMR_EMPTY_CART_MAX_SPEED_M_PER_SEC)
        self.acceleration = float(acceleration or AMR_ACCELERATION_M_PER_SEC2)
        self.deceleration = float(deceleration or AMR_DECELERATION_M_PER_SEC2)
        self.docking_speed = float(docking_speed or AMR_DOCKING_SPEED_M_PER_SEC)
        self.docking_distance = float(docking_distance if docking_distance is not None else AMR_DOCKING_DISTANCE_M)
        self.turn_time = float(turn_time if turn_time is not None else AMR_TURN_TIME_SEC)
        self.load_unload_time = float(load_unload_time)
        self.cmd = None
        self.amr_status = "UNDISPATCHED"

        self.route = None              # ACS가 보내준 전체 route
        self.raw_active_path = []      # ACS full path for the active leg
        self.active_visual_path = []   # full path used for smooth animation coordinates
        self.travel_distance = 0.0     # 현재 구간 이동 거리
        self.move_time = 0.0           # 현재 구간 MOVE 상태에서 걸리는 시간
        self.cell_size = MAP_CELL_SIZE_M  # map one-cell distance in meters
        self.cell_travel_time = AMR_CELL_TRAVEL_TIME

        self.move_start_time = None    # 전체 command 기준 이동 시작 시간, 필요 시 사용
        self.move_end_time = None      # 전체 command 기준 이동 종료 시간, 필요 시 사용
        self.delay_time = 0.0          # AMR 대기/지연 시간

        self.route_legs = []           # 이동 구간 목록, 필요 시 사용
        self.active_leg_index = 0      # 현재 몇 번째 구간인지, 필요 시 사용
        self.active_leg_name = None    # current_to_from / from_to / to_station
        self.active_path = None        # 현재 구간의 path

        self.leg_start_time = None     # 현재 구간 시작 시간
        self.leg_end_time = None       # 현재 구간 종료 시간
        self.leg_travel_distance = 0.0 # 현재 구간 이동 거리
        self.leg_move_time = 0.0       # 현재 구간 이동 시간
        self.leg_max_speed = self.speed
        self.active_motion_phases = []
        self.active_position_samples = []
        self.active_position_segments = []
        self.route_point_mode = AMR_ROUTE_POINT_MODE

        self.physical_log_interval = AMR_PHYSICAL_LOG_INTERVAL_SEC  # 주행 로그 출력 주기
        self.move_elapsed_time = 0.0      # 현재 구간에서 이미 이동한 시간
        self._last_move_ta = 0.0          # 이번 MOVE self-loop에서 진행된 시간

        self.leg_logs = []                # 구간별 주행 로그 저장

    def funcExternalTransition(self, strPort, event):
        state = self.getStateValue('state')
        if state == 'IDLE': # 작성
            if strPort == 'cmd_from_ACS':
                # 여러 AMR 중 이 AMR에게 온 명령인지 확인
                if event.AMR_id != self.AMR_id:
                    return
                self.cmd = event # ACS에서 오는 요청 정보

                self.amr_status = event.amr_status or "DISPATCHED"
                self.cmd.amr_status = self.amr_status


                # ACS가 보낸 route 정보 저장
                self.route = event.route

                # 첫 번째 이동 구간 설정: 현재 위치 -> from
                # 이 함수 안에서 active_leg_name, active_path, travel_distance, move_time이 설정됨
                self.set_active_leg(event.initial_leg or "current_to_from")

                # command 메시지 안에도 현재 구간 기준 계산 결과를 반영
                self.cmd.travel_distance = self.travel_distance
                self.cmd.speed = self.speed
                self.cmd.travel_time = self.move_time

                print(
                    f"AMR {self.AMR_id}: IDLE 상태에서 ACS command 수신, "
                    f"amr_status={self.amr_status}, "
                    f"active_leg={self.active_leg_name}, "
                    f"travel_distance={self.travel_distance}, "
                    f"speed={self.speed}, "
                    f"move_time={self.move_time}"
                )

                self.setStateValue('state', 'MOVE')

        elif state == 'MOVE':
            if strPort == 'cmd_from_ACS':
                # 1. 다른 AMR에게 배정된 command
                # broadcast로 들어온 것이므로 현재 AMR은 기존 이동 유지
                if event.AMR_id != self.AMR_id:
                    print(
                        f"AMR {self.AMR_id}: MOVE 상태에서 다른 AMR command 수신, "
                        f"현재 작업 계속 수행"
                    )
                    self.continueTimeAdvance()
                    return

                # 2. 자기 AMR_id command 수신
                # MOVE 상태라도 station 복귀 중인 UNDISPATCHED AMR이면 새 작업 배정 가능
                if self.amr_status == "UNDISPATCHED" and self.active_leg_name in ("to_station", "empty_to_station"):
                    print(
                        f"AMR {self.AMR_id}: MOVE(to_station) 중 자기 AMR_id command 수신, "
                        f"station 복귀 중단 후 새 작업 시작"
                    )

                    # 새 ACS command로 교체
                    self.cmd = event

                    # 새 메인 작업 시작이므로 DISPATCHED로 전환
                    self.amr_status = event.amr_status or "DISPATCHED"
                    self.cmd.amr_status = self.amr_status

                    # ACS가 보낸 새 route 저장
                    self.route = event.route

                    # 새 작업의 첫 번째 구간: 현재 위치 -> from
                    self.set_active_leg(event.initial_leg or "current_to_from")

                    # command 메시지 안에도 현재 구간 기준 계산 결과 반영
                    self.cmd.travel_distance = self.travel_distance
                    self.cmd.speed = self.speed
                    self.cmd.travel_time = self.move_time

                    print(
                        f"AMR {self.AMR_id}: 복귀 중 재배정 완료, "
                        f"amr_status={self.amr_status}, "
                        f"active_leg={self.active_leg_name}, "
                        f"travel_distance={self.travel_distance}, "
                        f"speed={self.speed}, "
                        f"move_time={self.move_time}"
                    )

                    self.setStateValue('state', 'MOVE')
                    return

                # 3. 자기 AMR_id command이지만 메인 작업 중인 경우
                # current_to_from / from_to 구간에서는 중복 배정으로 보고 기존 작업 유지
                print(
                    f"AMR {self.AMR_id}: MOVE 상태에서 자기 AMR_id command 수신, "
                    f"active_leg={self.active_leg_name}, "
                    f"amr_status={self.amr_status}. "
                    f"메인 작업 중 중복 배정 가능성이 있으므로 기존 작업 유지"
                )
                self.continueTimeAdvance()
                return

    def funcOutput(self):
        state = self.getStateValue('state')
        if state == 'IDLE':
            pass
        elif state == 'MOVE':
            physical_result_msg = self.make_physical_result_msg()
            self.addOutputEvent('physical_result_to_analyzer', physical_result_msg)
            self.addOutputEvent('physical_result_to_tm', physical_result_msg)

            print(
                f"AMR {self.AMR_id}: physical_result_to_analyzer and physical_result_to_tm 전송 - "
                # f"leg={self.active_leg_name}, "
                # f"elapsed={self.move_elapsed_time + self._last_move_ta}/{self.move_time}, "
                # f"task_id={physical_result_msg.task_id}, "
                # f"task_type={physical_result_msg.task_type}, "
                # f"from={physical_result_msg.from_}, "
                # f"to={physical_result_msg.to}, "
                # f"current_location={physical_result_msg.current_location}, "
                # f"travel_time={physical_result_msg.travel_time}, "
                # f"travel_distance={physical_result_msg.travel_distance}, "
                # f"speed={physical_result_msg.speed}, "
                # f"amr_status={physical_result_msg.amr_status}, "
                # f"result_status={physical_result_msg.result_status}"
            )

            # to_station 구간 이동이 끝난 경우:
            # station에 물리적으로 도착했으므로 TM에 완전 유휴/배정 가능 상태를 알림
            if self.active_leg_name in ("to_station", "empty_to_station") and self.is_move_complete_output():
                station_arrival_msg = self.make_station_arrival_msg()
                self.addOutputEvent('return_completed_to_tm', station_arrival_msg)

                print(
                    f"AMR {self.AMR_id}: MOVE(to_station) 완료 - "
                    f"station 도착 return_completed_to_tm 전송, "
                    f"current_location={station_arrival_msg.current_location}, "
                    f"amr_status={station_arrival_msg.amr_status}, "
                    f"idle_type={station_arrival_msg.idle_type}"
                )

        elif state == 'LOAD/UNLOAD': # 작성
            cmd_completed_msg = self.make_cmd_completed_msg()

            self.addOutputEvent('cmd_completed_to_mm', cmd_completed_msg)

            print(
                f"AMR {self.AMR_id}: LOAD/UNLOAD 상태에서 cmd_completed_to_mm 전송 - "
                # f"timestamp={cmd_completed_msg.timestamp}, "
                # f"leg={self.active_leg_name}, "
                # f"task_step={cmd_completed_msg.task_step}, "
                # f"from={cmd_completed_msg.from_}, "
                # f"to={cmd_completed_msg.to}, "
                # f"current_location={cmd_completed_msg.current_location}, "
                # f"current_situation={cmd_completed_msg.current_situation}, "
                # f"amr_status={cmd_completed_msg.amr_status}, "
                # f"load_type={cmd_completed_msg.load_type}, "
                # f"next_action={cmd_completed_msg.next_action}"
            )

        elif state == 'INFORM': # 작성 # amr 상태 undispatched 하다는 거 보내면 될 듯. 근데 이때 어느 작업 마치고 어디로 가는지도 보내야할 거 같기도
            inform_msg = self.make_return_completed_msg()
            self.addOutputEvent('return_completed_to_tm', inform_msg)
            print(
                f"AMR {self.AMR_id}: INFORM 상태에서 return_completed_to_tm 전송 - "
                f"AMR_id={inform_msg.AMR_id}, "
                f"from={inform_msg.from_}, "
                f"current_location={inform_msg.current_location}, "
                f"next_destination={inform_msg.next_destination}, "
                f"amr_status={inform_msg.amr_status}"
            )
        elif state == 'RECOVERY_WAIT':
            pass

    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'IDLE':
            pass
        elif state == 'MOVE': # 작성
            # 이번 MOVE self-loop에서 실제로 진행된 시간을 누적
            self.move_elapsed_time += self._last_move_ta

            # 아직 현재 구간 이동 시간이 남아 있으면 MOVE 상태 유지
            # 이 경우 다음 time advance 이후 physical_result_to_analyzer를 다시 보냄
            if self.move_elapsed_time < self.move_time:
                print(
                    f"AMR {self.AMR_id}: MOVE 진행 중 - "
                    f"leg={self.active_leg_name}, "
                    f"elapsed={self.move_elapsed_time}/{self.move_time} "
                    f"→ MOVE 상태 유지"
                )
                self.setStateValue('state', 'MOVE')
            
            else:
                # 현재 active_leg 구간 이동 완료
                if self.active_leg_name == "current_to_from":
                    # 현재 위치 -> from 이동 완료
                    # from 위치에서 LOAD 작업을 해야 하므로 LOAD/UNLOAD로 이동
                    print(
                        f"AMR {self.AMR_id}: MOVE 완료 - "
                        f"active_leg={self.active_leg_name}, "
                        f"from 위치 도착 → LOAD/UNLOAD 상태로 이동"
                    )
                    self.setStateValue('state', 'LOAD/UNLOAD')

                elif self.active_leg_name == "from_to":
                    # from -> to 이동 완료. 체인 회수가 있으면 계속 DISPATCHED.
                    self.amr_status = "DISPATCHED" if self.has_chain_recovery() else "UNDISPATCHED"
                    self.cmd.amr_status = self.amr_status

                    print(
                        f"AMR {self.AMR_id}: MOVE 완료 - "
                        f"active_leg={self.active_leg_name}, "
                        f"to 위치 도착, "
                        f"amr_status={self.amr_status} → LOAD/UNLOAD 상태로 이동"
                    )
                    self.setStateValue('state', 'LOAD/UNLOAD')  

                elif self.active_leg_name in ("to_recovery", "current_to_recovery"):
                    wait_time = self.recovery_wait_time()
                    if wait_time > 0:
                        self.cmd.recovery_wait_time = wait_time
                        print(
                            f"AMR {self.AMR_id}: 회수 위치 도착, "
                            f"wait_time={wait_time} → RECOVERY_WAIT"
                        )
                        self.setStateValue('state', 'RECOVERY_WAIT')
                    else:
                        self.cmd.recovery_wait_time = 0.0
                        self.setStateValue('state', 'LOAD/UNLOAD')

                elif self.active_leg_name == "recovery_to_empty":
                    self.amr_status = "DISPATCHED"
                    self.cmd.amr_status = self.amr_status
                    self.setStateValue('state', 'LOAD/UNLOAD')

                elif self.active_leg_name in ("to_station", "empty_to_station"):
                    # to -> station 이동 완료
                    # station 도착 완료 메시지는 MOVE output에서 이미 return_completed_to_tm으로 보냄
                    # AMR 배정 상태는 UNDISPATCHED로 두고 DEVS 상태만 IDLE로 이동
                    self.amr_status = "UNDISPATCHED"
                    self.cmd.amr_status = self.amr_status

                    print(
                        f"AMR {self.AMR_id}: MOVE 완료 - "
                        f"active_leg={self.active_leg_name}, "
                        f"station 도착 완료, "
                        f"amr_status={self.amr_status}, DEVS state=IDLE"
                    )
                    self.setStateValue('state', 'IDLE')      


        elif state == 'LOAD/UNLOAD': # 작성 # AMR 상태에 따라 분기 갈라지게 하면 될 듯
            amr_status = self.cmd.amr_status or self.amr_status

            # 디버깅용 데이터 임포트
            from_location = self.cmd.from_
            to_location = self.cmd.to
            goal_location = self.cmd.goal_location

            if self.active_leg_name == "current_to_from" and amr_status == "DISPATCHED":
                    print(
                        f"AMR {self.AMR_id}: LOAD/UNLOAD 완료, "
                        f"from={from_location}, to={to_location}, "
                        f"goal_location={goal_location}, "
                        f"amr_status={amr_status} → from_to MOVE 시작"
                    )

                    # current_to_from 이후에는 반드시 from_to 구간으로 바꿔야 함
                    self.set_active_leg("from_to")                    
                    self.setStateValue('state', 'MOVE')

            elif self.active_leg_name == "from_to" and amr_status == "UNDISPATCHED":
                print(
                    f"AMR {self.AMR_id}: LOAD/UNLOAD 완료, "
                    f"from={from_location}, to={to_location}, "
                    f"goal_location={goal_location}, "
                    f"amr_status={amr_status} → INFORM 상태로 이동"
                )
                self.setStateValue('state', 'INFORM')

            elif self.active_leg_name == "from_to" and self.has_chain_recovery():
                print(
                    f"AMR {self.AMR_id}: 공급 완료 후 체인 회수 구간 시작"
                )
                self.set_active_leg("to_recovery")
                self.setStateValue('state', 'MOVE')

            elif self.active_leg_name in ("to_recovery", "current_to_recovery"):
                print(
                    f"AMR {self.AMR_id}: 공대차 상차 완료, X 데포 이동 시작"
                )
                self.set_active_leg("recovery_to_empty")
                self.setStateValue('state', 'MOVE')

            elif self.active_leg_name == "recovery_to_empty":
                print(
                    f"AMR {self.AMR_id}: 공대차 반납 완료, station 복귀 시작"
                )
                self.set_active_leg("empty_to_station")
                self.setStateValue('state', 'MOVE')
            
        elif state == 'INFORM': 
            print(
                f"AMR {self.AMR_id}: INFORM 완료 - "
                f"메인 작업 완료 보고 후 to_station MOVE 시작"
            )

            # INFORM 이후에는 station 복귀 구간으로 이동해야 함
            self.set_active_leg("to_station")
            self.setStateValue('state', 'MOVE')

        elif state == 'RECOVERY_WAIT':
            print(f"AMR {self.AMR_id}: 회수 대기 완료 → LOAD/UNLOAD")
            self.setStateValue('state', 'LOAD/UNLOAD')

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'IDLE':
            return INF
        elif state == 'MOVE': 
            remaining_time = self.move_time - self.move_elapsed_time

            if remaining_time <= 0:
                self._last_move_ta = 0.0
                return 0

            self._last_move_ta = min(self.physical_log_interval, remaining_time)
            return self._last_move_ta
        
        elif state == 'LOAD/UNLOAD': 
            return self.load_unload_time
        elif state == 'INFORM': 
            return 0
        elif state == 'RECOVERY_WAIT':
            return self.recovery_wait_time()

    def funcSelect(self):
        return self



    def _point_from_location(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            coord = value.get("coordinates") or value
            if coord.get("x") is not None and coord.get("y") is not None:
                return int(coord["x"]), int(coord["y"])
        return None


    def calculate_path_distance(self, path, start=None):
        if path is None:
            return 0.0

        previous_point = self._point_from_location(start)
        distance = 0.0

        for point in path:
            current_point = self._point_from_location(point)
            if current_point is None:
                continue

            if previous_point is None:
                distance += self.cell_size
            else:
                cells = (
                    abs(current_point[0] - previous_point[0])
                    + abs(current_point[1] - previous_point[1])
                )
                distance += cells * self.cell_size

            previous_point = current_point

        return distance


    def calculate_path_time(self, path, start, speed):
        phases = self.build_motion_phases(path, start, speed)
        return phases[-1]["end_time"] if phases else 0.0


    def calculate_move_time(self, travel_distance, speed):
        if speed is None or speed <= 0:
            raise ValueError(
                f"AMR {self.AMR_id}: speed 값이 올바르지 않습니다: {speed}"
            )

        return self.speed_profile(travel_distance, speed)["total_time"]

    def speed_profile(self, distance, max_speed, start_speed=0.0, end_speed=0.0):
        distance = max(0.0, float(distance or 0.0))
        max_speed = float(max_speed or 0.0)
        start_speed = max(0.0, min(float(start_speed or 0.0), max_speed))
        end_speed = max(0.0, min(float(end_speed or 0.0), max_speed))
        if distance <= 0.0 or max_speed <= 0.0:
            return {
                "distance": distance,
                "max_speed": 0.0,
                "peak_speed": 0.0,
                "start_speed": 0.0,
                "end_speed": 0.0,
                "accel_time": 0.0,
                "cruise_time": 0.0,
                "decel_time": 0.0,
                "accel_distance": 0.0,
                "cruise_distance": 0.0,
                "decel_distance": 0.0,
                "total_time": 0.0,
                "shape": "none",
            }

        accel = max(self.acceleration, 1e-9)
        decel = max(self.deceleration, 1e-9)
        accel_distance = max(0.0, (max_speed ** 2 - start_speed ** 2) / (2 * accel))
        decel_distance = max(0.0, (max_speed ** 2 - end_speed ** 2) / (2 * decel))

        if distance >= accel_distance + decel_distance:
            cruise_distance = distance - accel_distance - decel_distance
            accel_time = max(0.0, (max_speed - start_speed) / accel)
            decel_time = max(0.0, (max_speed - end_speed) / decel)
            cruise_time = cruise_distance / max_speed
            peak_speed = max_speed
            shape = "trapezoid"
        else:
            peak_squared = (
                2 * distance
                + start_speed ** 2 / accel
                + end_speed ** 2 / decel
            ) / (1 / accel + 1 / decel)
            peak_speed = min(max_speed, max(start_speed, end_speed, peak_squared ** 0.5))
            accel_time = max(0.0, (peak_speed - start_speed) / accel)
            decel_time = max(0.0, (peak_speed - end_speed) / decel)
            cruise_time = 0.0
            accel_distance = max(0.0, (peak_speed ** 2 - start_speed ** 2) / (2 * accel))
            decel_distance = max(0.0, (peak_speed ** 2 - end_speed ** 2) / (2 * decel))
            cruise_distance = 0.0
            shape = "triangle"

        return {
            "distance": distance,
            "max_speed": max_speed,
            "peak_speed": peak_speed,
            "start_speed": start_speed,
            "end_speed": end_speed,
            "accel_time": accel_time,
            "cruise_time": cruise_time,
            "decel_time": decel_time,
            "accel_distance": accel_distance,
            "cruise_distance": cruise_distance,
            "decel_distance": decel_distance,
            "total_time": accel_time + cruise_time + decel_time,
            "shape": shape,
        }

    def profile_distance_at_time(self, profile, elapsed_time):
        elapsed_time = max(0.0, min(float(elapsed_time or 0.0), profile["total_time"]))
        accel_time = profile["accel_time"]
        cruise_time = profile["cruise_time"]
        accel = max(self.acceleration, 1e-9)
        decel = max(self.deceleration, 1e-9)
        start_speed = profile.get("start_speed", 0.0)
        peak_speed = profile["peak_speed"]

        if elapsed_time <= accel_time:
            return min(
                profile["distance"],
                start_speed * elapsed_time + 0.5 * accel * elapsed_time ** 2,
            )

        if elapsed_time <= accel_time + cruise_time:
            cruise_elapsed = elapsed_time - accel_time
            return min(
                profile["distance"],
                profile["accel_distance"] + peak_speed * cruise_elapsed,
            )

        decel_elapsed = elapsed_time - accel_time - cruise_time
        distance = (
            profile["accel_distance"]
            + profile["cruise_distance"]
            + peak_speed * decel_elapsed
            - 0.5 * decel * decel_elapsed ** 2
        )
        return min(profile["distance"], max(0.0, distance))

    def profile_speed_at_time(self, profile, elapsed_time):
        elapsed_time = max(0.0, min(float(elapsed_time or 0.0), profile["total_time"]))
        accel_time = profile["accel_time"]
        cruise_time = profile["cruise_time"]
        accel = max(self.acceleration, 1e-9)
        decel = max(self.deceleration, 1e-9)
        start_speed = profile.get("start_speed", 0.0)
        end_speed = profile.get("end_speed", 0.0)

        if elapsed_time <= accel_time:
            return min(profile["peak_speed"], start_speed + accel * elapsed_time)

        if elapsed_time <= accel_time + cruise_time:
            return profile["peak_speed"]

        decel_elapsed = elapsed_time - accel_time - cruise_time
        return max(end_speed, profile["peak_speed"] - decel * decel_elapsed)

    def build_motion_phases(self, path, start, max_speed):
        base_phases = self.build_base_motion_phases(path, start)
        docking_ranges = self.docking_ranges_for_leg(base_phases)

        pieces = []
        moving_cursor = 0.0

        for base_index, phase in enumerate(base_phases):
            if phase["kind"] != "drive":
                pieces.append(dict(phase))
                continue

            distance = phase["distance"]
            cuts = [0.0, distance]
            phase_start = moving_cursor
            phase_end = moving_cursor + distance
            for dock_start, dock_end in docking_ranges:
                for boundary in (dock_start, dock_end):
                    if phase_start < boundary < phase_end:
                        cuts.append(boundary - phase_start)
            cuts = sorted(set(round(cut, 10) for cut in cuts))

            for start_cut, end_cut in zip(cuts, cuts[1:]):
                part_distance = end_cut - start_cut
                if part_distance <= 0.0:
                    continue

                global_mid = moving_cursor + (start_cut + end_cut) / 2
                is_docking = any(
                    dock_start - 1e-9 <= global_mid <= dock_end + 1e-9
                    for dock_start, dock_end in docking_ranges
                )
                part_speed = min(max_speed, self.docking_speed) if is_docking else max_speed
                pieces.append({
                    "kind": "dock" if is_docking else "drive",
                    "base_index": base_index,
                    "distance": part_distance,
                    "max_speed": part_speed,
                })

            moving_cursor += distance

        phases = []
        time_cursor = 0.0
        distance_cursor = 0.0

        for index, piece in enumerate(pieces):
            if piece["kind"] not in ("drive", "dock"):
                duration = piece["duration"]
                phase = dict(piece)
                phase.update({
                    "start_time": time_cursor,
                    "end_time": time_cursor + duration,
                    "start_distance": distance_cursor,
                    "end_distance": distance_cursor,
                })
                phases.append(phase)
                time_cursor += duration
                continue

            previous_piece = pieces[index - 1] if index > 0 else None
            next_piece = pieces[index + 1] if index + 1 < len(pieces) else None

            start_speed = self.boundary_speed(previous_piece, piece)
            end_speed = self.boundary_speed(piece, next_piece)
            profile = self.speed_profile(
                piece["distance"],
                piece["max_speed"],
                start_speed=start_speed,
                end_speed=end_speed,
            )
            duration = profile["total_time"]
            phases.append({
                "kind": piece["kind"],
                "profile": profile,
                "duration": duration,
                "start_time": time_cursor,
                "end_time": time_cursor + duration,
                "start_distance": distance_cursor,
                "end_distance": distance_cursor + piece["distance"],
                "max_speed": piece["max_speed"],
                "start_speed": start_speed,
                "end_speed": end_speed,
            })
            time_cursor += duration
            distance_cursor += piece["distance"]

        return phases

    def boundary_speed(self, left_piece, right_piece):
        if not left_piece or not right_piece:
            return 0.0
        if left_piece.get("kind") not in ("drive", "dock"):
            return 0.0
        if right_piece.get("kind") not in ("drive", "dock"):
            return 0.0
        if left_piece.get("base_index") != right_piece.get("base_index"):
            return 0.0
        return min(left_piece.get("max_speed", 0.0), right_piece.get("max_speed", 0.0))

    def docking_ranges_for_leg(self, base_phases):
        dock = max(0.0, self.docking_distance)
        if dock <= 0.0:
            return []

        drive_ranges = []
        moving_cursor = 0.0
        for phase in base_phases:
            if phase["kind"] != "drive":
                continue
            distance = max(0.0, float(phase.get("distance", 0.0)))
            if distance <= 0.0:
                continue
            drive_ranges.append((moving_cursor, moving_cursor + distance))
            moving_cursor += distance

        if not drive_ranges:
            return []

        use_start, use_end = self.docking_sides_for_active_leg()
        ranges = []

        if use_start:
            first_start, first_end = drive_ranges[0]
            ranges.append((first_start, min(first_end, first_start + dock)))

        if use_end:
            last_start, last_end = drive_ranges[-1]
            ranges.append((max(last_start, last_end - dock), last_end))

        return [
            (start, end)
            for start, end in ranges
            if end - start > 1e-9
        ]

    def docking_sides_for_active_leg(self):
        if self.active_leg_name == "current_to_from":
            return False, True
        if self.active_leg_name in (
            "from_to",
            "current_to_recovery",
            "to_recovery",
            "recovery_to_empty",
        ):
            return True, True
        if self.active_leg_name in ("to_station", "empty_to_station"):
            return True, False
        return False, False

    def build_base_motion_phases(self, path, start):
        previous_point = self._point_from_location(start)
        active_direction = None
        active_distance = 0.0
        phases = []

        def flush_drive():
            nonlocal active_distance
            if active_distance > 0.0:
                phases.append({"kind": "drive", "distance": active_distance})
                active_distance = 0.0

        for point in path or []:
            current_point = self._point_from_location(point)
            if current_point is None:
                continue

            if previous_point is None:
                direction = None
                distance = self.cell_size
            elif current_point == previous_point:
                flush_drive()
                phases.append({"kind": "wait", "duration": 1.0})
                previous_point = current_point
                continue
            else:
                dx = current_point[0] - previous_point[0]
                dy = current_point[1] - previous_point[1]
                direction = self.direction(dx, dy)
                distance = (abs(dx) + abs(dy)) * self.cell_size

            if (
                active_distance > 0.0
                and direction is not None
                and active_direction is not None
                and direction != active_direction
            ):
                flush_drive()
                if self.turn_time > 0.0:
                    phases.append({"kind": "turn", "duration": self.turn_time})

            active_distance += distance
            if direction is not None:
                active_direction = direction
            previous_point = current_point

        flush_drive()
        return phases

    def direction(self, dx, dy):
        if dx == 0 and dy == 0:
            return None
        if abs(dx) >= abs(dy):
            return (1 if dx > 0 else -1, 0)
        return (0, 1 if dy > 0 else -1)

    def direction_between(self, from_point, to_point):
        if from_point is None or to_point is None:
            return None
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        return self.direction(dx, dy)

    def should_keep_motion_point(self, index, points, start_point):
        current = points[index]
        if current is None:
            return False

        previous_point = start_point if index == 0 else points[index - 1]
        next_point = points[index + 1] if index + 1 < len(points) else None

        if index == len(points) - 1:
            return True

        if current in CORE_AMR_PATH_POINTS:
            return True

        # Keep explicit waits so reservation delays remain in the physical timeline.
        if previous_point == current or next_point == current:
            return True

        previous_direction = self.direction_between(previous_point, current)
        next_direction = self.direction_between(current, next_point)
        return (
            previous_direction is not None
            and next_direction is not None
            and previous_direction != next_direction
        )

    def build_motion_path(self, path, start):
        raw_path = list(path or [])
        if self.route_point_mode == "full":
            return raw_path

        start_point = self._point_from_location(start)
        points = [self._point_from_location(point) for point in raw_path]
        compressed = []

        for index, raw_point in enumerate(raw_path):
            if self.should_keep_motion_point(index, points, start_point):
                compressed.append(raw_point)

        return compressed

    def build_position_samples(self, path, start):
        previous_point = self._point_from_location(start)
        cumulative = 0.0
        samples = []

        for raw_point in path or []:
            current_point = self._point_from_location(raw_point)
            if current_point is None:
                continue

            if previous_point is None:
                cumulative += self.cell_size
            elif current_point != previous_point:
                cells = (
                    abs(current_point[0] - previous_point[0])
                    + abs(current_point[1] - previous_point[1])
                )
                cumulative += cells * self.cell_size

            samples.append((cumulative, raw_point))
            previous_point = current_point

        return samples

    def build_position_segments(self, path, start):
        previous_point = self._point_from_location(start)
        cumulative = 0.0
        segments = []

        for raw_point in path or []:
            current_point = self._point_from_location(raw_point)
            if current_point is None:
                continue

            if previous_point is None:
                previous_point = current_point

            distance = (
                abs(current_point[0] - previous_point[0])
                + abs(current_point[1] - previous_point[1])
            ) * self.cell_size

            segments.append({
                "start_distance": cumulative,
                "end_distance": cumulative + distance,
                "from": previous_point,
                "to": current_point,
                "point": raw_point,
            })

            cumulative += distance
            previous_point = current_point

        return segments

    def _continuous_location(self, point):
        return {"x": float(point[0]), "y": float(point[1])}

    def location_at_distance_on_path(self, distance, leg_to):
        if not self.active_position_segments:
            if self.active_visual_path:
                return self.active_visual_path[-1]
            return leg_to

        distance = max(0.0, min(float(distance or 0.0), self.travel_distance))

        for segment in self.active_position_segments:
            if distance > segment["end_distance"]:
                continue

            start_distance = segment["start_distance"]
            end_distance = segment["end_distance"]
            length = end_distance - start_distance

            if length <= 0.0:
                return self._continuous_location(segment["to"])

            ratio = (distance - start_distance) / length
            from_x, from_y = segment["from"]
            to_x, to_y = segment["to"]
            return self._continuous_location((
                from_x + (to_x - from_x) * ratio,
                from_y + (to_y - from_y) * ratio,
            ))

        return self._continuous_location(self.active_position_segments[-1]["to"])

    def distance_at_elapsed_time(self, elapsed_time):
        elapsed_time = max(0.0, float(elapsed_time or 0.0))
        for phase in self.active_motion_phases:
            if elapsed_time > phase["end_time"]:
                continue
            if phase["kind"] not in ("drive", "dock"):
                return phase["start_distance"]

            local_time = elapsed_time - phase["start_time"]
            return phase["start_distance"] + self.profile_distance_at_time(
                phase["profile"],
                local_time,
            )
        return self.travel_distance

    def speed_at_elapsed_time(self, elapsed_time):
        elapsed_time = max(0.0, float(elapsed_time or 0.0))
        for phase in self.active_motion_phases:
            if elapsed_time > phase["end_time"]:
                continue
            if phase["kind"] not in ("drive", "dock"):
                return 0.0

            local_time = elapsed_time - phase["start_time"]
            return self.profile_speed_at_time(phase["profile"], local_time)
        return 0.0

    def make_physical_result_msg(self):  # MOVE 상태에서 주행 로그를 analyzer로 보낼 때 사용
        task_id = self.cmd.task_id
        task_type = self.cmd.task_type
        battery_level = self.cmd.battery_level
        move_num = self.cmd.move_num
        load_type = self.infer_load_type()
        carrying_type = self.infer_carrying_type()

        leg_from, leg_to = self.get_active_leg_locations()

        # funcOutput 시점에서는 아직 internalTransition 전이라
        # move_elapsed_time에 _last_move_ta가 더해지기 전이다.
        # 그래서 이번 output 기준의 진행 시간을 미리 계산한다.
        elapsed_time = self.move_elapsed_time + self._last_move_ta

        # 전체 구간 이동 시간을 넘지 않도록 보정
        if elapsed_time > self.move_time:
            elapsed_time = self.move_time

        # 현재까지 이동한 거리와 순간 속도는 가감속 프로파일 기준으로 계산
        elapsed_distance = self.distance_at_elapsed_time(elapsed_time)
        current_speed = self.speed_at_elapsed_time(elapsed_time)

        # 전체 구간 거리를 넘지 않도록 보정
        if elapsed_distance > self.travel_distance:
            elapsed_distance = self.travel_distance

        # 현재 위치 계산
        current_location = self.get_current_location_on_path(
            elapsed_time=elapsed_time,
            leg_to=leg_to
        )

        # 이번 output이 해당 구간의 마지막 output인지 확인
        is_complete = self.is_move_complete_output()

        if is_complete:
            result_status = "SUCCESS"
            event_description = (
                f"{self._location_name(leg_from)}에서 "
                f"{self._location_name(leg_to)}까지 주행 완료"
            )
        else:
            result_status = "RUNNING"
            event_description = (
                f"{self._location_name(leg_from)}에서 "
                f"{self._location_name(leg_to)}까지 주행 중"
            )

        msg = Message(
            product=self.cmd.product,
            part=self.cmd.part,
            cart_count=self.cmd.cart_count,
            generated_time=self.cmd.generated_time,
            AMR_id=self.AMR_id,
            task_id=task_id,
            task_type=task_type,
            task_step=self.active_leg_name,

            from_=leg_from,
            to=leg_to,

            # 이동 중이면 중간 위치, 완료 시점이면 leg_to
            current_location=current_location,

            # 현재 active leg의 route
            route=self.active_path,

            start_time=self.leg_start_time,
            end_time=self.getTime(),
            timestamp=self.getTime(),

            # 전체 move_time이 아니라 현재까지 진행된 시간/거리
            travel_time=elapsed_time,
            travel_distance=elapsed_distance,

            speed=current_speed,
            delay_time=self.delay_time,

            battery_level=battery_level,
            amr_status=self.amr_status,
            result_status=result_status,
            load_type=load_type,
            carrying_type=carrying_type,

            move_num=move_num,
            event_description=event_description
        )

        return msg
    
    def get_current_location_on_path(self, elapsed_time, leg_to):
        if not self.active_position_segments:
            return leg_to

        if self.move_time <= 0:
            return self.location_at_distance_on_path(0.0, leg_to)

        if self.is_move_complete_output():
            return leg_to

        traveled_distance = self.distance_at_elapsed_time(elapsed_time)
        return self.location_at_distance_on_path(traveled_distance, leg_to)
    

    
    def _location_name(self, location):
        if location is None:
            return None

        if isinstance(location, dict):
            return (
                location.get("location_id")
                or location.get("location_code")
                or location.get("name")
                or str(location)
            )

        return str(location)

    def make_return_completed_msg(self): # INFORM 상태에서 메인 작업 완료 후 station으로 복귀하러 간다는 정보를 TM에 보낼 때 사용
        from_location = self.cmd.from_
        to_location = self.cmd.to
        station_location = self.cmd.goal_location

        # INFORM은 from -> to 메인 작업이 끝난 시점
        # 아직 station으로 이동하기 전이므로 현재 위치는 to
        current_location = to_location

        # 메인 작업은 끝났으므로 AMR은 작업 배정 관점에서 UNDISPATCHED 상태
        self.amr_status = "UNDISPATCHED"
        self.current_location = current_location
        self.cmd.amr_status = self.amr_status

        msg = Message(
            AMR_id=self.AMR_id,

            goal_type="MAIN",
            goal_location=to_location,

            from_=from_location,
            to=to_location,

            current_location=current_location,
            arrival_event=f"{self._location_name(current_location)} 도착",

            idle_type="RETURN_IDLE",
            amr_status=self.amr_status,

            next_destination=station_location,
            next_action=f"{self._location_name(station_location)} 이동",

            timestamp=self.getTime(),

            event_description=(
                f"AMR {self.AMR_id}: 메인 작업 완료. "
                f"{self._location_name(from_location)}에서 "
                f"{self._location_name(to_location)}까지 작업 완료 후 "
                f"{self._location_name(station_location)}로 이동 예정"
            )
        )

        return msg
    
    def make_station_arrival_msg(self):  # MOVE(to_station)가 끝났을 때 station에 물리적으로 도착했다는 정보를 TM에 보낼 때 사용
        to_location, station_location = self.get_active_leg_locations()
        if station_location is None:
            station_location = self.cmd.goal_location
        if to_location is None:
            to_location = self.cmd.to

        # station에 물리적으로 도착했으므로 AMR 배정 상태는 UNDISPATCHED로 유지
        self.amr_status = "UNDISPATCHED"
        self.current_location = station_location
        self.cmd.amr_status = self.amr_status

        msg = Message(
            AMR_id=self.AMR_id,

            goal_type="STATION",
            goal_location=station_location,

            from_=to_location,
            to=station_location,

            # MOVE(to_station)이 끝난 시점이므로 현재 위치는 station
            current_location=station_location,

            arrival_event=f"{self._location_name(station_location)} 도착",
            idle_type="STATION_IDLE",

            amr_status=self.amr_status,

            next_destination=None,
            next_action=None,

            timestamp=self.getTime(),

            event_description=(
                f"AMR {self.AMR_id} station 도착 완료. "
                f"dispatching 가능 상태"
            )
        )

        return msg
    
    def max_speed_for_active_leg(self):
        if self.active_leg_name == "recovery_to_empty":
            return self.empty_cart_max_speed
        if self.active_leg_name == "from_to":
            if self.is_empty_cart_route():
                return self.empty_cart_max_speed
            return self.loaded_max_speed
        return self.unloaded_max_speed

    def is_empty_cart_route(self):
        if self.active_leg_name == "recovery_to_empty":
            return True
        from_name = self._location_name(getattr(self.cmd, "from_", None)) or ""
        to_name = self._location_name(getattr(self.cmd, "to", None)) or ""
        recovery_from = self._location_name(getattr(self.cmd, "recovery_from", None)) or ""
        recovery_to = self._location_name(getattr(self.cmd, "recovery_to", None)) or ""
        text = f"{from_name} {to_name} {recovery_from} {recovery_to}"
        return "Recall" in text or "_R" in text

    def set_active_leg(self, leg_name): # 현재 이동할 구간을 정하고 그 구간의 이동 거리와 이동 시간 계산함수
        self.active_leg_name = leg_name

        if self.route is None:
            self.raw_active_path = []
        else:
            self.raw_active_path = list(self.route.get(leg_name, []) or [])

        if self.raw_active_path is None:
            self.raw_active_path = []

        self.leg_start_time = self.getTime()

        leg_from, _ = self.get_active_leg_locations()
        start_point = leg_from
        if self.active_leg_name in ("current_to_from", "current_to_recovery"):
            start_point = getattr(self.cmd, "current_coord", None) or leg_from

        self.active_path = self.build_motion_path(self.raw_active_path, start_point)
        self.active_visual_path = list(self.raw_active_path)

        self.leg_max_speed = self.max_speed_for_active_leg()
        self.speed = self.leg_max_speed
        self.active_motion_phases = self.build_motion_phases(
            self.active_visual_path,
            start_point,
            self.leg_max_speed,
        )
        self.active_position_samples = self.build_position_samples(
            self.active_visual_path,
            start_point,
        )
        self.active_position_segments = self.build_position_segments(
            self.active_visual_path,
            start_point,
        )
        self.leg_travel_distance = self.calculate_path_distance(
            self.active_visual_path,
            start_point,
        )
        self.leg_move_time = (
            self.active_motion_phases[-1]["end_time"]
            if self.active_motion_phases
            else 0.0
        )

        # 기존 변수도 현재 구간 기준으로 맞춰둠
        self.travel_distance = self.leg_travel_distance
        self.move_time = self.leg_move_time
        # 새 구간 시작이므로 이동 누적 시간 초기화
        self.move_elapsed_time = 0.0
        self._last_move_ta = 0.0        

        print(
            f"AMR {self.AMR_id}: active_leg 설정 - "
            f"leg={self.active_leg_name}, "
            f"path_points={len(self.raw_active_path)}->{len(self.active_path)} summary, "
            f"distance={self.travel_distance}, "
            f"max_speed={self.leg_max_speed}, "
            f"move_time={self.move_time}, "
            f"start_time={self.leg_start_time}"
        )

    def get_active_leg_locations(self): # 주행 로그에서 구간별로 (current to from, from to, to station) 찍을 수 있게해줌
        current_location = self.cmd.current_location
        from_location = self.cmd.from_
        to_location = self.cmd.to
        station_location = self.cmd.goal_location
        recovery_from = self.cmd.recovery_from or from_location
        recovery_to = self.cmd.recovery_to or to_location

        if self.active_leg_name == "current_to_from":
            return current_location, from_location

        elif self.active_leg_name == "from_to":
            return from_location, to_location

        elif self.active_leg_name == "current_to_recovery":
            return current_location, recovery_from

        elif self.active_leg_name == "to_recovery":
            return to_location, recovery_from

        elif self.active_leg_name == "recovery_to_empty":
            return recovery_from, recovery_to

        elif self.active_leg_name == "empty_to_station":
            return recovery_to, station_location

        elif self.active_leg_name == "to_station":
            return to_location, station_location

        return None, None
    
    def is_move_complete_output(self):
        return self.move_elapsed_time + self._last_move_ta >= self.move_time
    

    def make_cmd_completed_msg(self):
        # ACS command에서 기본 작업 정보 꺼내기
        cmd_current_location = self.cmd.current_location
        cmd_from_location = self.cmd.from_
        cmd_to_location = self.cmd.to
        station_location = self.cmd.goal_location

        task_id = self.cmd.task_id
        task_type = self.cmd.task_type
        product = self.cmd.product
        part = self.cmd.part
        cart_count = self.cmd.cart_count
        generated_time = self.cmd.generated_time
        move_num = self.cmd.move_num

        load_type = self.infer_load_type()

        if self.active_leg_name == "current_to_from":
            # 현재 위치 -> from 위치 이동 완료 후 LOAD 작업 수행
            # 아직 from -> to 작업이 남아 있으므로 DISPATCHED 상태 유지
            self.amr_status = "DISPATCHED"
            self.cmd.amr_status = self.amr_status

            leg_from = cmd_current_location
            leg_to = cmd_from_location

            current_location = leg_to
            current_situation = f"{self._location_name(current_location)} 도착 및 LOAD 완료"

            next_destination = cmd_to_location
            next_action = f"{self._location_name(next_destination)} 이동"

            result_status = "LOAD_COMPLETED"

            event_description = (
                f"AMR {self.AMR_id}: "
                f"{self._location_name(leg_from)}에서 "
                f"{self._location_name(leg_to)}까지 이동 후 LOAD 완료. "
                f"다음 목적지={self._location_name(next_destination)}"
            )

        elif self.active_leg_name == "from_to":
            # from 위치 -> to 위치 이동 완료 후 UNLOAD 작업 수행
            # 체인 회수가 있으면 계속 DISPATCHED, 아니면 기존처럼 작업 완료 상태로 전환
            self.amr_status = "DISPATCHED" if self.has_chain_recovery() else "UNDISPATCHED"
            self.cmd.amr_status = self.amr_status

            leg_from = cmd_from_location
            leg_to = cmd_to_location

            current_location = leg_to
            current_situation = f"{self._location_name(current_location)} 도착 및 UNLOAD 완료"

            next_destination = station_location
            next_action = f"{self._location_name(next_destination)} 이동"

            result_status = "UNLOAD_COMPLETED"

            event_description = (
                f"AMR {self.AMR_id}: "
                f"{self._location_name(leg_from)}에서 "
                f"{self._location_name(leg_to)}까지 이동 후 UNLOAD 완료. "
                f"메인 작업 완료 후 {self._location_name(next_destination)}로 복귀 예정"
            )

        elif self.active_leg_name in ("to_recovery", "current_to_recovery"):
            self.amr_status = "DISPATCHED"
            self.cmd.amr_status = self.amr_status

            leg_from = cmd_current_location if self.active_leg_name == "current_to_recovery" else cmd_to_location
            leg_to = self.cmd.recovery_from or cmd_from_location
            current_location = leg_to
            current_situation = f"{self._location_name(current_location)} 도착 및 LOAD 완료"
            next_destination = self.cmd.recovery_to or cmd_to_location
            next_action = f"{self._location_name(next_destination)} 이동"
            result_status = "LOAD_COMPLETED"
            event_description = (
                f"AMR {self.AMR_id}: "
                f"{self._location_name(leg_from)}에서 "
                f"{self._location_name(leg_to)}까지 이동 후 공대차 LOAD 완료."
            )

        elif self.active_leg_name == "recovery_to_empty":
            self.amr_status = "DISPATCHED"
            self.cmd.amr_status = self.amr_status

            leg_from = self.cmd.recovery_from or cmd_from_location
            leg_to = self.cmd.recovery_to or cmd_to_location
            current_location = leg_to
            current_situation = f"{self._location_name(current_location)} 도착 및 UNLOAD 완료"
            next_destination = station_location
            next_action = f"{self._location_name(next_destination)} 이동"
            result_status = "UNLOAD_COMPLETED"
            event_description = (
                f"AMR {self.AMR_id}: "
                f"{self._location_name(leg_from)}에서 "
                f"{self._location_name(leg_to)}까지 이동 후 공대차 UNLOAD 완료."
            )

        msg = Message(
            product=product,
            part=part,
            cart_count=cart_count,
            generated_time=generated_time,
            AMR_id=self.AMR_id,
            task_id=task_id,
            task_type=task_type,
            task_step=self.active_leg_name,

            from_=leg_from,
            to=leg_to,

            current_location=current_location,
            current_situation=current_situation,

            amr_status=self.amr_status,
            result_status=result_status,
            load_type=load_type,
            chain_recovery=self.cmd.chain_recovery,
            recovery_from=self.cmd.recovery_from,
            recovery_to=self.cmd.recovery_to,
            recovery_ready_time=self.cmd.recovery_ready_time,
            recovery_wait_time=self.cmd.recovery_wait_time,

            next_destination=next_destination,
            next_action=next_action,

            move_num=move_num,
            timestamp=self.getTime(),

            event_description=event_description
        )

        return msg
    
    def infer_load_type(self):
        if self.active_leg_name in ("to_recovery", "current_to_recovery", "recovery_to_empty"):
            return "공대차"

        from_location = self.cmd.from_
        to_location = self.cmd.to

        from_name = self._location_name(from_location)
        to_name = self._location_name(to_location)

        text = f"{from_name} {to_name}"

        # 회수데포 -> 자재데포 흐름이면 공대차
        if "회수" in text or "Recall" in text or "_R" in text:
            return "공대차"

        # 자재데포 -> 공정데포 흐름이면 실대차
        return "실대차"

    def infer_carrying_type(self):
        if self.active_leg_name == "recovery_to_empty":
            return "공대차"
        if self.active_leg_name == "from_to":
            if self.is_empty_cart_route():
                return "공대차"
            return "실대차"
        return "미적재"

    def has_chain_recovery(self):
        return bool(
            self.cmd
            and self.cmd.chain_recovery
            and self.cmd.recovery_from is not None
            and self.cmd.recovery_to is not None
        )

    def recovery_wait_time(self):
        ready_time = getattr(self.cmd, "recovery_ready_time", None)
        if ready_time is None:
            return 0.0
        try:
            return max(0.0, float(ready_time) - float(self.getTime()))
        except (TypeError, ValueError):
            return 0.0
