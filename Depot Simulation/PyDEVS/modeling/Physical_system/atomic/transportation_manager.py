import json

from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
from config import MAP_PATH

INF = float('inf')


class TransportationManager(DEVSAtomicModel):

    def __init__(self, ID, objConfiguration=None):
        super().__init__(ID)
        self.objConfiguration = objConfiguration

        # AMR_LIST 읽기
        self.amr_configs = []
        if self.objConfiguration is not None:
            self.amr_configs = self.objConfiguration.getConfiguration("AMR_LIST") or []

        self.addStateVariable('state', 'WAIT')

        self.addInputPort('amr_req_from_ACS')
        self.addInputPort('amr_update_from_ACS')
        self.addInputPort('return_completed_from_amr')
        self.addInputPort('physical_result_from_amr')

        self.addOutputPort('amr_info_to_ACS')

        self.active_job = None
        self.map_data = self._load_map()
        self.ps_nodes = self.map_data.get('ps_nodes', [])
        self.location_index = self._build_location_index()
        self.amr_fleet = self._initialize_amr_fleet()
        self.physical_results_by_amr = {}

    def funcExternalTransition(self, strPort, event):
        state = self.getStateValue('state')
        message = self._to_message(event)
        self._debug(f"input {strPort} while {state}: {self._describe_message(message)}")

        if strPort == 'physical_result_from_amr':
            self._store_physical_result(message)
            self._update_amr_live_position(message)
            self.setStateValue('state','WAIT')
            return

        if strPort == 'return_completed_from_amr':
            self._update_amr_status(message)
            if state == 'WAIT':
                self.active_job = self._job(message)
                self.setStateValue('state', 'INFO')
            else:
                self.setStateValue('state', state)
            return

        if state == 'WAIT':
            if strPort == 'amr_req_from_ACS':
                self.active_job = self._job(message)
                self.setStateValue('state', 'INFO')

            elif strPort == 'amr_update_from_ACS':
                self._assign_amr(message)
                self.setStateValue('state','WAIT')

    def funcOutput(self):
        state = self.getStateValue('state')

        if state == 'INFO':
            update_message = self.active_job.get('update_message')
            if update_message is not None:
                self._update_amr_status(update_message)

            amr_list = self._amr_info_messages(self.active_job['request'])

            self._debug(
                "output amr_info_to_ACS: "
                f"count={len(amr_list)}"
            )

            msg = Message(
                amr_list=amr_list,
                event_description="TM sends available AMR list to ACS"
            )

            self.addOutputEvent('amr_info_to_ACS', msg)

    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            pass
        elif state == 'INFO':
            self.active_job = None
            self.setStateValue('state', 'WAIT')

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            return INF
        elif state == 'INFO':
            return 0

    def funcSelect(self):
        return self

    def _to_message(self, event):
        """입력 이벤트를 Message 복사본으로 정규화한다."""
        if isinstance(event, Message):
            return event.copy()

        data = dict(event)
        if data.get('AMR_id') is None and data.get('amr_id') is not None:
            data['AMR_id'] = data['amr_id']
        return Message.from_dict(data)

    def _load_map(self):
        """modeling/map의 map JSON을 읽어 TM 내부 메모리로 보관한다."""
        with MAP_PATH.open(encoding='utf-8') as file:
            return json.load(file)

    def _build_location_index(self):
        """WH/ML/PS 위치를 location_id 기준으로 빠르게 찾기 위한 index를 만든다."""
        location_index = {}
        for collection_name in ('wh_locations', 'ml_nodes', 'ps_nodes'):
            for node in self.map_data.get(collection_name, []):
                location_id = node.get('location_id')
                if location_id is not None:
                    location_index[location_id] = node
        return location_index

    def _initialize_amr_fleet(self):
        """AMR_LIST와 PS map을 연결해 TM의 AMR 상태 메모리를 만든다."""
        amr_fleet = []
        for config in self.amr_configs:
            current_location = config.get('current_location')
            ps_node = self._ps_node(current_location)
            if ps_node is not None:
                ps_node['occupied'] = 1

            amr_fleet.append({
                'AMR_id': config.get('AMR_id'),
                'amr_product': config.get('product'),
                'current_location': current_location,
                'location_id': current_location,
                'battery_level': config.get('battery_level', 100.0),
                'amr_type': self._amr_type(current_location, config),
                'amr_status': 'UNDISPATCHED',
                'idle_type': 'STATION_IDLE' if ps_node is not None and ps_node.get('occupied') == 1 else None,
                'status': 'parking' if ps_node is not None and ps_node.get('occupied') == 1 else None,
            })
        return amr_fleet

    def _job(self, request, update_message=None):
        """INFO 상태에서 처리할 요청과 선택적 AMR 상태 업데이트 메시지를 묶는다."""
        job = {
            'request': request,
        }
        if update_message is not None:
            job['update_message'] = update_message
        return job

    def _amr_info_messages(self, request):
        """요청 메시지에 가용 AMR별 상태 정보를 얹은 응답을 만든다."""
        messages = [
            request.update(values=self._amr_payload(amr))
            for amr in self._idle_amrs()
        ]
        self._debug(f"available AMRs selected: count={len(messages)}")
        return messages

    def _idle_amrs(self):
        """PS에 주차 중이거나 복귀 중이라 ACS가 선택 가능한 AMR만 골라낸다."""
        return [
            amr
            for amr in self.amr_fleet
            if self._is_idle_amr(amr)
        ]

    def _is_idle_amr(self, amr):
        """AMR이 가용 상태인지 판단한다."""
        if amr.get('amr_status') == 'DISPATCHED':
            return False

        if amr.get('idle_type') in ('RETURN_IDLE', 'STATION_IDLE'):
            return True

        ps_node = self._ps_node(amr.get('current_location'))
        if ps_node is not None and ps_node.get('occupied') == 1:
            return True

        return False

    def _update_amr_status(self, message):
        """AMR 완료/복귀 메시지로 TM의 AMR 상태 메모리를 갱신한다."""
        amr_id = self._message_value(message, 'AMR_id')
        if amr_id is None:
            return

        amr = self._find_amr(amr_id)
        if amr is None:
            amr = {'AMR_id': amr_id}
            self.amr_fleet.append(amr)
            self._debug(f"new AMR registered: {amr_id}")

        for key in (
            'location_id',
            'location_code',
            'node',
            'x',
            'y',
            'access_node_id',
            'current_location',
            'goal_location',
            'next_destination',
            'amr_status',
            'idle_type',
            'battery_level',
            'amr_type',
            'amr_product',
        ):
            value = self._message_value(message, key)
            if value is not None:
                amr[key] = value

        status = self._message_value(message, 'status')
        if status is not None:
            amr['status'] = status

        current_location = self._location_id(amr.get('current_location'))
        if current_location is not None:
            amr['current_location'] = current_location
            amr['location_id'] = current_location

        idle_type = amr.get('idle_type')
        if idle_type == 'STATION_IDLE':
            self._mark_station_arrival(amr)
        elif idle_type == 'RETURN_IDLE':
            amr['status'] = 'returning'

        self._debug(f"AMR status updated: {amr}")

    def _assign_amr(self, message):
        """Mark the AMR selected by ACS as DISPATCHED."""
        amr_id = self._message_value(message, 'AMR_id')
        if amr_id is None:
            return

        amr = self._find_amr(amr_id)
        if amr is None:
            self._debug(f"assign skipped, AMR not found: {amr_id}")
            return

        ps_node = self._ps_node(amr.get('current_location'))
        if ps_node is not None:
            ps_node['occupied'] = 0
            self._debug(f"PS node occupied updated: {ps_node.get('location_id')} -> 0")

        amr['amr_status'] = self._message_value(message, 'amr_status') or 'DISPATCHED'
        amr['idle_type'] = None
        amr['status'] = self._message_value(message, 'status') or 'working'
        self._debug(f"AMR assigned to DISPATCHED: {amr_id}")

    def _find_amr(self, amr_id):
        """AMR_id로 TM 내부 AMR 상태 record를 찾는다."""
        for amr in self.amr_fleet:
            if amr.get('AMR_id') == amr_id:
                return amr
        return None

    def _store_physical_result(self, message):
        """Keep the latest physical result event by AMR_id."""
        amr_id = self._message_value(message, 'AMR_id')
        if amr_id is None:
            return
        self.physical_results_by_amr[amr_id] = message.copy()

    def _update_amr_live_position(self, message):
        """Reflect AMR movement events into TM's AMR state."""
        amr_id = self._message_value(message, 'AMR_id')
        if amr_id is None:
            return

        amr = self._find_amr(amr_id)
        if amr is None:
            amr = {'AMR_id': amr_id}
            self.amr_fleet.append(amr)

        current_location = self._message_value(message, 'current_location')
        location_id = self._location_id(current_location)
        x, y = self._position_from_location(current_location)

        if location_id is not None:
            amr['current_location'] = location_id
            amr['location_id'] = location_id
        elif x is not None and y is not None:
            amr['current_location'] = {'x': x, 'y': y}
            amr['location_id'] = None

        if x is not None:
            amr['x'] = x
        if y is not None:
            amr['y'] = y

        for key in ('amr_status', 'battery_level', 'next_destination', 'goal_location'):
            value = self._message_value(message, key)
            if value is not None:
                amr[key] = value

        if amr.get('amr_status') == 'DISPATCHED':
            amr['idle_type'] = None
            amr['status'] = 'working'
        elif (
            amr.get('amr_status') == 'UNDISPATCHED'
            and self._message_value(message, 'task_step') == 'to_station'
            and self._message_value(message, 'result_status') == 'SUCCESS'
        ):
            self._mark_station_arrival(amr)

        self._debug(
            "AMR live position updated: "
            f"AMR_id={amr_id}, "
            f"current_location={amr.get('current_location')}, "
            f"x={amr.get('x')}, y={amr.get('y')}, "
            f"amr_status={amr.get('amr_status')}"
        )

    def _amr_payload(self, amr):
        """AMR 상태 record를 Message.update()에 넣을 payload로 변환한다."""
        current_location_value = amr.get('current_location') or amr.get('location_id')
        current_location = self._location_id(current_location_value)
        location = self.location_index.get(current_location, {})
        coordinates = location.get('coordinates') or {}
        x = coordinates.get('x', amr.get('x'))
        y = coordinates.get('y', amr.get('y'))
        payload_current_location = current_location if current_location is not None else current_location_value

        return {
            'AMR_id': amr.get('AMR_id'),
            'location_id': current_location,
            'location_code': current_location,
            'node': location.get('slot', amr.get('node')),
            'x': x,
            'y': y,
            'access_node_id': location.get('access_node_id', amr.get('access_node_id')),
            'current_location': payload_current_location,
            'next_destination': amr.get('next_destination'),
            'amr_status': amr.get('amr_status'),
            'idle_type': amr.get('idle_type'),
            'status': amr.get('status'),
            'amr_type': location.get('amr_type', amr.get('amr_type')),
            'amr_product': amr.get('amr_product'),
            'battery_level': amr.get('battery_level'),
        }

    def _mark_station_arrival(self, amr):
        """AMR이 PS에 도착했을 때 해당 ps_node occupied를 1로 바꾼다."""
        station_id = self._location_id(
            amr.get('current_location')
            or amr.get('goal_location')
            or amr.get('next_destination')
        )
        ps_node = self._ps_node(station_id)
        if ps_node is None:
            return

        ps_node['occupied'] = 1
        amr['current_location'] = ps_node.get('location_id')
        amr['location_id'] = ps_node.get('location_id')
        amr['amr_type'] = ps_node.get('amr_type', amr.get('amr_type'))
        amr['amr_status'] = 'UNDISPATCHED'
        amr['idle_type'] = 'STATION_IDLE'
        amr['status'] = 'parking'
        self._debug(f"PS node occupied updated: {ps_node.get('location_id')} -> 1")

    def _ps_node(self, location_id):
        """location_id가 PS 노드이면 해당 ps_node를 반환한다."""
        location_id = self._location_id(location_id)
        if location_id is None:
            return None
        for node in self.ps_nodes:
            if node.get('location_id') == location_id:
                return node
        return None

    def _location_id(self, value):
        """문자열/dict 위치값에서 location_id를 추출한다."""
        if isinstance(value, dict):
            return value.get('location_id') or value.get('location_code')
        if isinstance(value, str):
            return value
        return None

    def _position_from_location(self, value):
        if isinstance(value, str):
            node = self.location_index.get(value, {})
            coordinates = node.get('coordinates') or {}
            return coordinates.get('x'), coordinates.get('y')

        if isinstance(value, dict):
            coordinates = value.get('coordinates') or value
            return coordinates.get('x'), coordinates.get('y')

        return None, None

    def _amr_type(self, location_id, config):
        """AMR 타입을 config 또는 PS node에서 읽는다."""
        if config.get('amr_type') is not None:
            return config.get('amr_type')
        ps_node = self._ps_node(location_id)
        if ps_node is not None:
            return ps_node.get('amr_type')
        return None

    def _message_value(self, message, *keys):
        """Message에서 첫 번째 non-None 필드값을 읽는다."""
        for key in keys:
            attr = 'from_' if key == 'from' else key
            if not hasattr(message, attr):
                continue
            value = getattr(message, attr)
            if value is not None:
                return value
        return None

    def _debug(self, text):
        """TransportationManager 디버깅 로그를 일정한 형식으로 출력한다."""
        print(f"[{self.getTime():08.2f}][TransportationManager][DEBUG] {text}")

    def _describe_message(self, message):
        """디버깅용으로 AMR 메시지의 핵심 필드를 문자열로 만든다."""
        return (
            f"AMR_id={self._message_value(message, 'AMR_id')}, "
            f"location_id={self._message_value(message, 'location_id')}, "
            f"amr_status={self._message_value(message, 'amr_status')}, "
            f"idle_type={self._message_value(message, 'idle_type')}"
        )
