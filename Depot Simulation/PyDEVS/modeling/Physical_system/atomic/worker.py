import json

from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
from config import MAP_PATH, PART_WORK_TIME

INF = float('inf')


class Worker(DEVSAtomicModel):

    def __init__(self, ID, work_time_by_part=None, default_work_time=5):
        super().__init__(ID)

        self.addStateVariable('state', 'WAIT')

        self.addInputPort('arrive_from_mm')
        self.addInputPort('m_info_from_mm')

        self.addOutputPort('item_depot_update_to_MCS')
        self.addOutputPort('w_req_to_mm')

        self.work_time_by_part = dict(work_time_by_part or PART_WORK_TIME)
        self.default_work_time = default_work_time
        self.active_jobs = {}
        self.waiting_jobs_by_location = {}
        self.completed_waiting_for_recovery = []
        self.pending_recovery_requests = []
        self.pending_updates = []
        self.last_update_time = 0.0
        self.map_data = self._load_json('map.json')
        self.recovery_depots = self._build_recovery_depots(self.map_data)
        self.recovery_overflow_queues = {}
        self.recovery_bottleneck_current = {}
        self.recovery_bottleneck_total = {}

    def funcExternalTransition(self, strPort, event):
        self._advance_jobs_to_now()
        self._debug(
            f"input {strPort} while {self.getStateValue('state')}: "
            f"{self._describe_event(event)}"
        )

        if strPort == 'arrive_from_mm':
            for arrival in self._to_messages(event):
                self._start_work(arrival)

        elif strPort == 'm_info_from_mm':
            self._handle_mm_info(event)

        self._refresh_state()

    def funcOutput(self):
        state = self.getStateValue('state')
        if state == 'UPDATE':
            msg = Message(
                item_list=[self.pending_updates[0]],
                event_description="Worker sends item depot updates to MCS"
            )
            self._debug(
                "output item_depot_update_to_MCS: "
                f"{self._describe_event(msg.item_list)}"
            )
            self.addOutputEvent('item_depot_update_to_MCS', msg)

        elif state == 'REQUEST':
            msg = self.pending_recovery_requests[0]
            self._debug(
                "output waiting recovery w_req_to_mm: "
                f"{self._describe_message(msg)}"
            )
            self.addOutputEvent('w_req_to_mm', msg)

        elif state == 'WORK':
            for key, job in self._completed_job_items():
                msg = self._prepare_recovery_request(job)
                if msg is None:
                    self._debug(
                        "completed worker job waits for recovery depot: "
                        f"location={key}, {self._describe_message(job['arrival'])}"
                    )
                    continue
                self._debug(
                    "output w_req_to_mm: "
                    f"{self._describe_message(msg)}"
                )
                self.addOutputEvent('w_req_to_mm', msg)

    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'UPDATE':
            if self.pending_updates:
                self.pending_updates.pop(0)
        elif state == 'REQUEST':
            if self.pending_recovery_requests:
                self.pending_recovery_requests.pop(0)
        elif state == 'WORK':
            self._advance_jobs_to_now()
            for key in self._completed_job_keys():
                self._debug(f"completed worker job removed: location={key}")
                job = self.active_jobs.pop(key, None)
                if job is not None and job.get('request_message') is None:
                    self.completed_waiting_for_recovery.append(job['arrival'])
                self._start_next_waiting_job(key)

        self._refresh_state()

    def funcTimeAdvance(self):
        if self.pending_updates:
            return 0
        if self.pending_recovery_requests:
            return 0
        if self.active_jobs:
            return min(
                max(0, job['remaining_time'])
                for job in self.active_jobs.values()
            )
        return INF

    def funcSelect(self):
        return self

    def _load_json(self, file_name):
        """Physical system data 폴더에서 JSON 데이터를 읽는다."""
        with MAP_PATH.open(encoding='utf-8') as file:
            return json.load(file)

    def _build_recovery_depots(self, map_data):
        """map.json의 ML R 노드만 회수데포 목록으로 구성한다."""
        recovery_depots = []
        for node in map_data.get('ml_nodes', []):
            if node.get('rack_type', node.get('node_type')) != 'R':
                continue
            recovery_depots.append(self._recovery_depot(node))
        return recovery_depots

    def _recovery_depot(self, source):
        """map.json의 회수데포 노드를 Worker 내부 위치 record로 변환한다."""
        coordinates = source.get('coordinates', {})
        node_index = source.get('slot', source.get('node_index'))
        return {
            'location_id': source.get('location_id'),
            'location_code': source.get('location_id'),
            'product': source.get('product'),
            'part': source.get('part', source.get('station')),
            'node': f"R{node_index}",
            'x': coordinates.get('x'),
            'y': coordinates.get('y'),
            'access_node_id': source.get('access_node_id'),
            'type': '회수데포',
            'status': self._occupied_status(source.get('occupied')),
        }

    def _occupied_status(self, occupied):
        """map의 occupied 값을 메시지 status 문자열로 변환한다."""
        return 'occupied' if occupied else 'empty'

    def _start_work(self, arrival):
        """P1/P2 도착 이벤트를 공정데포 위치별 FIFO 작업으로 등록한다."""
        key = self._process_location_key(arrival)

        if key in self.active_jobs:
            self.waiting_jobs_by_location.setdefault(key, []).append(arrival)
            self._debug(
                "queued worker job: "
                f"location={key}, queue_len={len(self.waiting_jobs_by_location[key])}, "
                f"{self._describe_message(arrival)}"
            )
            return

        self._begin_work(key, arrival)

    def _begin_work(self, key, arrival):
        """공정데포 위치별 active 작업을 하나 시작한다."""
        self.active_jobs[key] = {
            'arrival': arrival,
            'request_message': None,
            'remaining_time': self._part_work_time(arrival),
        }
        self._debug(
            "started worker job: "
            f"location={key}, slot={self._process_slot(arrival)}, "
            f"remaining_time={self.active_jobs[key]['remaining_time']}"
        )
        return True

    def _start_next_waiting_job(self, key):
        """완료된 공정데포 위치의 FIFO 대기 작업을 즉시 시작한다."""
        queue = self.waiting_jobs_by_location.get(key)
        while queue:
            arrival = queue.pop(0)
            if not queue:
                self.waiting_jobs_by_location.pop(key, None)
            if self._begin_work(key, arrival):
                return
            queue = self.waiting_jobs_by_location.get(key)

    def _process_location_key(self, arrival):
        """같은 공정데포 노드의 작업을 FIFO로 처리하기 위한 key를 만든다."""
        return (
            self._location_id(self._message_value(arrival, 'current_location', 'to'))
            or self._message_value(arrival, 'location_id')
            or self._message_value(arrival, 'part')
        )

    def _handle_mm_info(self, event):
        """MM의 응답/회수데포 상태 갱신을 Worker 내부 상태에 반영한다."""
        message = self._to_message(event)
        if self._message_value(message, 'info_type') == 'recovery_update':
            self._update_recovery_depot(message)
            self._recover_completed_waiting_jobs()
            return

        update_message = self._update_message(message)
        if update_message is not None:
            self.pending_updates.append(update_message)
        else:
            self._debug("no valid update message created")

    def _update_message(self, event):
        """MM이 알려준 공대차 데포 위치를 MCS update 메시지로 만든다."""
        message = self._to_message(event)
        if message.from_ is None or message.to is None:
            return None

        return message.update(values={'info_type': 'update'})

    def _to_messages(self, event):
        """단일/리스트 입력 이벤트를 Message 리스트로 정규화한다."""
        if isinstance(event, list):
            return [self._to_message(entry) for entry in event]

        message = self._to_message(event)
        return [message]

    def _to_message(self, event):
        """입력 이벤트 하나를 Message 복사본으로 정규화한다."""
        if isinstance(event, Message):
            return event.copy()

        data = dict(event)
        if data.get('product') is None and data.get('proudct') is not None:
            data['product'] = data['proudct']
        return Message.from_dict(data)

    def _request_message(self, arrivals):
        """작업 완료 후 MM에 보낼 회수데포 출발 위치 요청 메시지를 만든다."""
        for arrival in arrivals:
            request_message = self._request_message_for_arrival(arrival)
            if request_message is not None:
                return request_message
        return None

    def _request_message_for_arrival(self, arrival):
        """작업 완료를 MM에 알린다. 오븐 c만 즉시 일반 회수 요청을 만든다."""
        process_payload = self._arrival_payload(arrival)
        process_payload['status'] = 'empty_cart_ready'
        process_payload['process_status'] = 'empty_cart_ready'
        completed_time = self.getTime()

        values = {
            **process_payload,
            'from': process_payload,
            'to': None,
            'current_location': process_payload,
            'status': 'empty_cart_ready',
            'process_status': 'empty_cart_ready',
            'process_completed_time': completed_time,
            'timestamp': completed_time,
        }

        if self._is_oven_c(arrival):
            return arrival.update(
                values={
                    **values,
                    'task_id': f"{self._message_value(arrival, 'task_id')}_REC",
                    'task_type': 'recovery',
                    'info_type': 'request',
                    'initial_leg': 'current_to_recovery',
                    'chain_recovery': False,
                    'event_description': 'Oven c process completed; recovery requested',
                }
            )

        return arrival.update(
            values={
                **values,
                'info_type': 'process_completed',
                'event_description': 'Process completed; empty cart waits at process depot',
            }
        )

    def _make_recovery_request(
        self,
        recovery_depot,
        arrival,
        info_type='request',
        bottleneck_values=None,
    ):
        """회수데포에서 공대차 자재데포로 보낼 요청 메시지를 만든다."""
        recovery_depot['status'] = 'occupied'
        self._debug(
            "selected recovery depot as from: "
            f"{recovery_depot.get('location_id')}"
        )
        recovery_payload = self._recovery_payload(recovery_depot, arrival)
        values = {
            **recovery_payload,
            'info_type': info_type,
            'from': recovery_payload,
            'to': None,
            'timestamp': self.getTime(),
        }
        if bottleneck_values:
            values.update(bottleneck_values)

        return arrival.update(values=values)

    def _queue_recovery_overflow(self, recovery_depot, arrival):
        """R이 이미 점유 중이면 완료품을 R 병목 대기열에 넣고 MM에 알린다."""
        location_id = recovery_depot.get('location_id')
        if location_id is None:
            return None

        self.recovery_overflow_queues.setdefault(location_id, []).append(arrival)
        current = self.recovery_bottleneck_current.get(location_id, 0) + 1
        total = self.recovery_bottleneck_total.get(location_id, 0) + 1
        self.recovery_bottleneck_current[location_id] = current
        self.recovery_bottleneck_total[location_id] = total
        recovery_depot['status'] = 'occupied'
        self._debug(
            "queued recovery overflow: "
            f"location={location_id}, current={current}, total={total}"
        )

        return self._make_recovery_request(
            recovery_depot,
            arrival,
            info_type='recovery_overflow',
            bottleneck_values=self._bottleneck_values(
                recovery_depot,
                '회수데포',
                '미완료_진입',
                current,
                total,
                delta_current=1,
                delta_total=1,
            ),
        )

    def _work_time(self, arrivals):
        """도착 건별 part에 맞춘 작업 시간을 계산한다."""
        return sum(
            self._part_work_time(arrival)
            for arrival in arrivals
        )

    def _part_work_time(self, arrival):
        """도착 메시지의 part에 맞는 작업 시간을 읽는다."""
        product = self._message_value(arrival, 'product')
        part = self._message_value(arrival, 'part')
        if product is not None and (product, part) in self.work_time_by_part:
            return self.work_time_by_part[(product, part)]
        return self.work_time_by_part.get(part, self.default_work_time)

    def _is_oven_c(self, message):
        return (
            self._message_value(message, 'product') == 'B'
            and self._message_value(message, 'part') == 'c'
        )

    def _advance_jobs_to_now(self):
        """마지막 갱신 시각 이후 흐른 시간만큼 모든 part 작업의 남은 시간을 줄인다."""
        now = self._now()
        elapsed = now - self.last_update_time
        if elapsed <= 0:
            self.last_update_time = now
            return

        for job in self.active_jobs.values():
            job['remaining_time'] = max(0, job['remaining_time'] - elapsed)
        self.last_update_time = now

    def _now(self):
        """엔진 현재 시각을 안전하게 읽는다."""
        engine = getattr(self, 'engine', None)
        if engine not in (None, -1) and hasattr(engine, 'getTime'):
            return engine.getTime()
        return getattr(self, 'time', 0)

    def _completed_jobs(self):
        """현재 time advance에서 완료되는 part 작업들을 반환한다."""
        return [job for _, job in self._completed_job_items()]

    def _completed_job_items(self):
        """현재 time advance에서 완료되는 part 작업 key/job들을 반환한다."""
        if not self.active_jobs:
            return []

        min_remaining = min(job['remaining_time'] for job in self.active_jobs.values())
        return [
            (key, job)
            for key, job in self.active_jobs.items()
            if job['remaining_time'] <= min_remaining + 1e-9
        ]

    def _completed_job_keys(self):
        """남은 시간이 0인 완료 작업 key 목록을 반환한다."""
        return [
            key
            for key, job in self.active_jobs.items()
            if job['remaining_time'] <= 1e-9
        ]

    def _refresh_state(self):
        """대기 update가 있으면 우선 출력하고, 아니면 작업 상태를 유지한다."""
        if self.pending_updates:
            self.setStateValue('state', 'UPDATE')
        elif self.pending_recovery_requests:
            self.setStateValue('state', 'REQUEST')
        elif self.active_jobs:
            self.setStateValue('state', 'WORK')
        else:
            self.setStateValue('state', 'WAIT')

    def _prepare_recovery_request(self, job):
        """완료 job에 대해 빈 R이 있으면 회수 요청을 한 번만 만든다."""
        if job.get('request_message') is not None:
            return job['request_message']

        request_message = self._request_message_for_arrival(job['arrival'])
        if request_message is None:
            return None

        job['request_message'] = request_message
        return request_message

    def _recover_completed_waiting_jobs(self):
        """R이 빈 뒤 회수 대기 완료품을 FIFO로 다시 요청한다."""
        remaining = []
        for arrival in self.completed_waiting_for_recovery:
            request_message = self._request_message_for_arrival(arrival)
            if request_message is None:
                remaining.append(arrival)
                continue
            self.pending_recovery_requests.append(request_message)
        self.completed_waiting_for_recovery = remaining

    def _update_recovery_depot(self, message):
        """MM이 알려준 R 상태를 반영하고, R 병목 대기열을 하나씩 회수로 넘긴다."""
        location_id = self._message_value(message, 'location_id', 'location_code')
        if location_id is None:
            location_id = self._location_id(self._message_value(message, 'from', 'to', 'current_location'))
        if location_id is None:
            return

        status = self._message_value(message, 'status')
        if status is None:
            result_status = self._message_value(message, 'result_status')
            if result_status == 'LOAD_COMPLETED':
                status = 'empty'
            elif result_status == 'UNLOAD_COMPLETED':
                status = 'occupied'
        if status is None:
            return

        for recovery_depot in self.recovery_depots:
            if recovery_depot.get('location_id') != location_id:
                continue
            recovery_depot['status'] = status
            self._debug(
                "recovery depot status updated: "
                f"{location_id} -> {status}"
            )
            if status == 'empty':
                self._resume_recovery_overflow(recovery_depot)
            return

    def _resume_recovery_overflow(self, recovery_depot):
        """R에서 한 대가 빠진 뒤 대기 중인 완료품을 다음 회수 대상으로 올린다."""
        location_id = recovery_depot.get('location_id')
        queue = self.recovery_overflow_queues.get(location_id) or []
        if not queue:
            return

        arrival = queue.pop(0)
        if not queue:
            self.recovery_overflow_queues.pop(location_id, None)

        current = max(0, self.recovery_bottleneck_current.get(location_id, 0) - 1)
        total = self.recovery_bottleneck_total.get(location_id, 0)
        self.recovery_bottleneck_current[location_id] = current
        recovery_depot['status'] = 'occupied'
        self._debug(
            "recovery overflow moved to pickup: "
            f"location={location_id}, current={current}, total={total}"
        )

        self.pending_recovery_requests.append(
            self._make_recovery_request(
                recovery_depot,
                arrival,
                info_type='recovery_resume',
                bottleneck_values=self._bottleneck_values(
                    recovery_depot,
                    '회수데포',
                    '대기품_회수시작',
                    current,
                    total,
                    delta_current=-1,
                    delta_total=0,
                ),
            )
        )

    def _find_recovery_depot(self, arrival, require_empty=True):
        """도착 메시지의 product/part에 맞는 회수데포를 찾는다."""
        product = self._message_value(arrival, 'product')
        part = self._message_value(arrival, 'part')

        candidates = []
        for recovery_depot in self.recovery_depots:
            if product is not None and recovery_depot.get('product') != product:
                continue
            if part is not None and recovery_depot.get('part') != part:
                continue
            if require_empty and recovery_depot.get('status') != 'empty':
                continue
            candidates.append(recovery_depot)

        if not candidates:
            return None

        if require_empty:
            return candidates[0]

        return sorted(
            candidates,
            key=lambda depot: (
                self.recovery_bottleneck_current.get(depot.get('location_id'), 0),
                str(depot.get('location_id') or ''),
            ),
        )[0]

    def _recovery_payload(self, recovery_depot, arrival):
        """회수데포 위치와 대차 part 수량을 출력 payload로 만든다."""
        return {
            'location_id': recovery_depot.get('location_id'),
            'location_code': recovery_depot.get('location_code'),
            'product': recovery_depot.get('product'),
            'part': recovery_depot.get('part'),
            'node': recovery_depot.get('node'),
            'x': recovery_depot.get('x'),
            'y': recovery_depot.get('y'),
            'access_node_id': recovery_depot.get('access_node_id'),
            'type': recovery_depot.get('type'),
            'capacity': self._message_value(arrival, 'capacity', 'cart_count'),
            'status': recovery_depot.get('status'),
        }

    def _bottleneck_values(
        self,
        depot,
        depot_type,
        event_type,
        current,
        total,
        delta_current,
        delta_total,
    ):
        """MM/Analyzer/visualizer가 공통으로 읽는 병목 이벤트 필드를 만든다."""
        return {
            'bottleneck_event': True,
            'bottleneck_event_type': event_type,
            'bottleneck_depot_type': depot_type,
            'bottleneck_current': current,
            'bottleneck_total': total,
            'bottleneck_delta_current': delta_current,
            'bottleneck_delta_total': delta_total,
        }

    def _arrival_payload(self, arrival):
        """공정데포 도착 메시지를 ACS 작업의 from 위치 payload로 만든다."""
        return {
            'location_id': self._message_value(arrival, 'location_id'),
            'location_code': self._message_value(arrival, 'location_code', 'location_id'),
            'product': self._message_value(arrival, 'product'),
            'part': self._message_value(arrival, 'part'),
            'node': self._message_value(arrival, 'node'),
            'x': self._message_value(arrival, 'x'),
            'y': self._message_value(arrival, 'y'),
            'access_node_id': self._message_value(arrival, 'access_node_id'),
            'type': self._message_value(arrival, 'type'),
            'capacity': self._message_value(arrival, 'capacity', 'cart_count'),
            'status': self._message_value(arrival, 'status'),
        }

    def _process_slot(self, message):
        """P1/P2 위치의 slot 번호를 읽는다."""
        node = self._message_value(message, 'node')
        if node is not None:
            try:
                return int(node)
            except (TypeError, ValueError):
                pass

        for key in ('current_location', 'to', 'location_id'):
            value = self._message_value(message, key)
            location_id = self._location_id(value) if key != 'location_id' else value
            if not isinstance(location_id, str):
                continue
            suffix = location_id.rsplit('_P', 1)
            if len(suffix) == 2:
                try:
                    return int(suffix[1])
                except ValueError:
                    return None
        return None

    def _location_id(self, value):
        """dict/string 위치값에서 location_id를 추출한다."""
        if isinstance(value, dict):
            return value.get('location_id')
        return value

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
        """Worker 디버깅 로그를 일정한 형식으로 출력한다."""
        print(f"[{self.getTime():08.2f}][Worker][DEBUG] {text}")

    def _describe_message(self, message):
        """디버깅용으로 Worker 입력 메시지의 핵심 필드를 문자열로 만든다."""
        return (
            f"product={self._message_value(message, 'product')}, "
            f"part={self._message_value(message, 'part')}, "
            f"capacity={self._message_value(message, 'capacity', 'cart_count')}, "
            f"location_id={self._message_value(message, 'location_id')}"
        )

    def _describe_event(self, event):
        """단일/리스트 이벤트를 디버깅 문자열로 만든다."""
        if isinstance(event, list):
            return f"count={len(event)}"
        if isinstance(event, Message):
            return self._describe_message(event)
        if isinstance(event, dict):
            return (
                f"product={event.get('product')}, "
                f"part={event.get('part')}, "
                f"location_id={event.get('location_id')}"
            )
        return str(event)
