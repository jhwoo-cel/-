import json
from copy import deepcopy

from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
from config import MAP_PATH, PART_WORK_TIME

INF = float('inf')


class ManufacturingManager(DEVSAtomicModel):

    def __init__(self, ID):
        super().__init__(ID)

        self.addStateVariable('state', 'WAIT')

        self.addInputPort('m_req_from_MCS')
        self.addInputPort('cmd_completed_from_amr')
        self.addInputPort('item_update_from_gen')
        self.addInputPort('w_req_from_wk')

        self.addOutputPort('m_info_to_MCS')
        self.addOutputPort('work_info_to_analyzer')
        self.addOutputPort('arrive_to_wk')
        self.addOutputPort('m_info_to_wk')

        self.active_job = None
        self.map_data = self._load_map()
        self.reserved_from_location_ids = set()
        self.reserved_to_location_ids = set()
        self.empty_cart_fallback_indexes = {}
        self.input_queue = []
        self.process_bottleneck_current = {}
        self.process_bottleneck_total = {}
        self.empty_cart_bottleneck_current = {}
        self.empty_cart_bottleneck_total = {}
        self.process_state_by_location = {}
        self.last_supplied_process_by_key = {}
        self.next_process_slot_by_key = {}
        self.reserved_process_recovery_location_ids = set()
        self._initialize_process_states()

    def funcExternalTransition(self, strPort, event):
        state = self.getStateValue('state')
        message = self._to_message(event)
        self._debug(
            f"input {strPort} while {state}: {self._describe_message(message)}"
        )

        self.input_queue.append((strPort, message))
        if state == 'WAIT':
            self._start_next_job()
        else:
            self.continueTimeAdvance()

    def funcOutput(self):
        state = self.getStateValue('state')
        message = self.active_job['message']

        if state == 'INFO':
            part = self._value(message, 'part')
            print(
                "[ManufacturingManage] output to MCS: "
                f"{message.info_type} for {message.product}/{part}"
            )
            print('===================================')
            print(message.from_)
            print(message.to)
            print('===================================')
            self.addOutputEvent('m_info_to_MCS', message)

        elif state == 'RESULT':
            self._debug(f"output work_info_to_analyzer: {self._describe_event(message)}")
            self.addOutputEvent('work_info_to_analyzer', message)
            for mcs_message in self.active_job.get('mcs_messages', []):
                self._debug(
                    "output m_info_to_MCS: "
                    f"{mcs_message.info_type} "
                    f"for {mcs_message.product}/{mcs_message.part}, "
                    f"to={self._location_id(mcs_message.to)}"
                )
                self.addOutputEvent('m_info_to_MCS', mcs_message)
            for worker_message in self.active_job.get('worker_messages', []):
                self._debug(
                    "output recovery update to Worker: "
                    f"{self._describe_message(worker_message)}"
                )
                self.addOutputEvent('m_info_to_wk', worker_message)

        elif state == 'SIGNAL':
            self._debug(f"output arrive_to_wk: {self._describe_event(message)}")
            self.addOutputEvent('arrive_to_wk', message)

        elif state == 'PARK':
            for work_message in self.active_job.get('work_messages', []):
                self._debug(
                    "output bottleneck work_info_to_analyzer: "
                    f"{work_message.get('bottleneck_event_type')} "
                    f"location={work_message.get('location_id')} "
                    f"current={work_message.get('bottleneck_current')}, "
                    f"total={work_message.get('bottleneck_total')}"
                )
                self.addOutputEvent('work_info_to_analyzer', work_message)
            if self._value(message, 'initial_leg') == 'current_to_recovery':
                self._debug(
                    "output direct recovery m_info_to_MCS: "
                    f"from={self._location_id(message.from_)}, "
                    f"to={self._location_id(message.to)}"
                )
                self.addOutputEvent('m_info_to_MCS', message)
            else:
                self._debug(
                    "output m_info_to_wk: "
                    f"from={self._location_id(message.from_)}, "
                    f"to={self._location_id(message.to)}"
                )
                self.addOutputEvent('m_info_to_wk', message)
            for release_message in self.active_job.get('release_messages', []):
                self._debug(
                    "output m_info_to_MCS: "
                    f"{release_message.info_type} "
                    f"for {release_message.product}/{release_message.part}, "
                    f"to={self._location_id(release_message.to)}"
                )
                self.addOutputEvent('m_info_to_MCS', release_message)

    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            pass
        elif state == 'INFO':
            self.active_job = None
            self.setStateValue('state', 'WAIT')
            self._start_next_job()
        elif state == 'RESULT':
            signal_message = self.active_job.get('signal_message')
            if signal_message is not None:
                self.active_job = self._job('SIGNAL', signal_message)
                self.setStateValue('state', 'SIGNAL')
            else:
                self.active_job = None
                self.setStateValue('state', 'WAIT')
                self._start_next_job()
        elif state == 'SIGNAL':
            self.active_job = None
            self.setStateValue('state', 'WAIT')
            self._start_next_job()
        elif state == 'PARK':
            selected_record = self.active_job.get('selected_record')
            if selected_record is not None:
                selected_record['occupied'] = 1
                self._debug(
                    "empty cart depot occupied after response: "
                    f"{selected_record.get('location_id')}"
                )
            self.active_job = None
            self.setStateValue('state', 'WAIT')
            self._start_next_job()

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            return INF
        elif state == 'INFO':
            return 0
        elif state == 'RESULT':
            return 0
        elif state == 'SIGNAL':
            return 0
        elif state == 'PARK':
            return 0
        return INF

    def funcSelect(self):
        return self

    def _load_map(self):
        """Physical system data 폴더에서 map.json을 읽는다."""
        with MAP_PATH.open(encoding='utf-8') as file:
            return json.load(file)

    def _initialize_process_states(self):
        for record in self.map_data.get('ml_nodes', []):
            if record.get('rack_type') != 'P':
                continue
            status = 'empty' if record.get('occupied') == 0 else 'processing'
            record['process_status'] = status
            record['process_completed_time'] = None
            self.process_state_by_location[record.get('location_id')] = {
                'status': status,
                'ready_time': None,
                'record': deepcopy(record),
            }

    def _to_message(self, event):
        """입력 이벤트를 Message로 정규화하고 part 기준으로 맞춘다."""
        if isinstance(event, Message):
            message = event.copy()
        else:
            data = dict(event)
            if data.get('AMR_id') is None and data.get('amr_id') is not None:
                data['AMR_id'] = data['amr_id']
            if data.get('AMR_id') is None and data.get('AGV_id') is not None:
                data['AMR_id'] = data['AGV_id']
            if data.get('product') is None and data.get('proudct') is not None:
                data['product'] = data['proudct']
            message = Message.from_dict(data)
        return message

    def _job(self, state, message, signal_message=None, **extra):
        """출력 상태와 출력 메시지를 active_job 형태로 묶는다."""
        job = {
            'state': state,
            'message': message,
        }
        if signal_message is not None:
            job['signal_message'] = signal_message
        job.update(extra)
        return job

    def _start_next_job(self):
        """대기 입력을 하나씩 꺼내 기존 상태 전이 job으로 변환한다."""
        while self.input_queue:
            strPort, message = self.input_queue.pop(0)

            if strPort == 'm_req_from_MCS':
                self.active_job = self._job(
                    'INFO',
                    self._m_info_message(message, 'response'),
                )
                self.setStateValue('state', 'INFO')
                return

            if strPort == 'item_update_from_gen':
                updated_locations = self._update_item_map(message)
                self.active_job = self._job(
                    'INFO',
                    self._m_info_message(
                        message,
                        'update',
                        from_location=self._first_record(updated_locations),
                    ),
                )
                self.setStateValue('state', 'INFO')
                return

            if strPort == 'cmd_completed_from_amr':
                self.active_job = self._result_job(message)
                self.setStateValue('state', self.active_job['state'])
                return

            if strPort == 'w_req_from_wk':
                if self._value(message, 'info_type') == 'process_completed':
                    self.active_job = self._process_completed_job(message)
                else:
                    self.active_job = self._park_job(message)
                self.setStateValue('state', self.active_job['state'])
                return

        self.active_job = None
        self.setStateValue('state', 'WAIT')

    def _m_info_message(self, message, info_type, from_location=None):
        """MCS에 보낼 from/to 후보를 선택해 Message로 반환한다."""
        if from_location is None:
            from_location = self._select_from_location(message)
        to_location = self._select_to_location(message)
        chain_values = self._chain_recovery_values(message, to_location)

        if info_type == 'response' and from_location is not None and to_location is not None:
            self.reserved_from_location_ids.add(from_location.get('location_id'))
            self.reserved_to_location_ids.add(to_location.get('location_id'))
            self._debug(
                "reserved schedule: "
                f"from={self._location_id(from_location)}, "
                f"to={self._location_id(to_location)}"
            )
        else:
            self._debug(
                "m_info selected without reservation: "
                f"info_type={info_type}, "
                f"from={self._location_id(from_location)}, "
                f"to={self._location_id(to_location)}"
            )

        return message.update(
            values={
                'info_type': info_type,
                'from': from_location,
                'to': to_location,
                **chain_values,
            }
        )

    def _update_item_map(self, message):
        """Generator part 업데이트를 WH 실대차 위치 occupied=1로 반영한다."""
        empty_locations = self._records(
            'wh_locations',
            message,
            rack_type='O',
            occupied=0,
            copy_record=False,
        )
        cart_count = self._positive_int(self._value(message, 'cart_count'))
        if cart_count is None:
            cart_count = 1

        updated_locations = sorted(empty_locations, key=self._record_sort_key)[:cart_count]
        updated = []
        for location in updated_locations:
            location['occupied'] = 1
            updated.append(location.get('location_id'))

        self._debug(
            "part map updated: "
            f"product={self._value(message, 'product')}, "
            f"part={self._value(message, 'part')}, "
            f"locations={updated}"
        )
        return updated_locations

    def _result_job(self, message):
        """AMR 완료 입력으로 analyzer 출력과 필요 시 worker 신호를 만든다."""
        process_occupied_before = self._arrival_occupied_by_location(message, '공정데포')
        work_messages = self._work_info_messages(message)
        work_messages.extend(self._empty_cart_bottleneck_messages(message))
        process_recovery_result = self._handle_process_recovery_pickup(message)
        work_messages.extend(process_recovery_result['work_messages'])
        arrival_result = self._process_arrival_messages(
            message,
            process_occupied_before,
        )
        work_messages.extend(arrival_result['work_messages'])
        mcs_messages = (
            process_recovery_result['mcs_messages']
            + arrival_result['mcs_messages']
        )
        recovery_updates = self._recovery_update_messages(message)
        return self._job(
            'RESULT',
            work_messages,
            signal_message=arrival_result['signal_message'],
            mcs_messages=mcs_messages,
            worker_messages=recovery_updates,
        )

    def _process_completed_job(self, message):
        """Worker가 알려준 P 작업 완료를 반영하고 필요하면 즉시 회수를 요청한다."""
        location_id = self._process_location_id_from_worker_request(message)
        if (
            location_id is not None
            and location_id not in self.reserved_process_recovery_location_ids
        ):
            return self._park_job(message)

        self._mark_process_completed(message)
        return self._job(
            'RESULT',
            [],
            signal_message=None,
            mcs_messages=[],
            worker_messages=[],
        )

    def _park_job(self, message):
        """Worker 요청에 맞는 WH 공대차 데포 위치를 선택해 응답 job을 만든다."""
        request_from_process = self._request_from_process_location(message)
        if request_from_process:
            self._mark_process_completed(message)
        release_result = (
            {'release_messages': [], 'work_messages': []}
            if request_from_process
            else self._release_process_location(message)
        )
        work_messages = list(release_result.get('work_messages', []))
        self._occupy_recovery_location(message)
        if self._value(message, 'bottleneck_event'):
            work_messages.append(self._bottleneck_work_info_from_message(message))

        if self._value(message, 'info_type') == 'recovery_overflow':
            return self._job(
                'PARK',
                message.update(values={'info_type': 'response', 'to': None}),
                release_messages=release_result['release_messages'],
                work_messages=work_messages,
            )

        empty_cart_record = self._select_empty_cart_location(message)
        if empty_cart_record is None:
            return self._job(
                'PARK',
                message.update(values={'info_type': 'response', 'to': None}),
                release_messages=release_result['release_messages'],
                work_messages=work_messages,
            )

        process_location = self._value(message, 'from')
        to_location = self._empty_cart_location_payload(empty_cart_record, message)
        if request_from_process:
            process_location_id = self._process_location_id_from_worker_request(message)
            if process_location_id is not None:
                self.reserved_process_recovery_location_ids.add(process_location_id)

        response_values = {
            'info_type': 'response',
            'from': process_location,
            'to': to_location,
            'task_type': 'recovery',
            'initial_leg': 'current_to_recovery',
            'chain_recovery': False,
        }
        if request_from_process:
            response_values.update({
                'current_location': process_location,
                'recovery_from': process_location,
                'recovery_to': to_location,
            })

        response = message.update(values=response_values)
        self._debug(
            "selected empty cart depot: "
            f"from={self._location_id(response.from_)}, "
            f"to={self._location_id(response.to)}, "
            f"occupied_before={empty_cart_record.get('occupied')}"
        )
        return self._job(
            'PARK',
            response,
            selected_record=empty_cart_record,
            release_messages=release_result['release_messages'],
            work_messages=work_messages,
        )

    def _release_process_location(self, message):
        """Worker가 작업 완료 후 P에서 작업물을 빼면 해당 공정데포를 빈 상태로 바꾼다."""
        result = {
            'release_messages': [],
            'work_messages': [],
        }
        if self._value(message, 'info_type') == 'recovery_resume':
            return result
        if self._request_from_process_location(message):
            return result

        process_location_id = self._process_location_id_from_worker_request(message)
        if process_location_id is None:
            return result

        for record in self.map_data.get('ml_nodes', []):
            if record.get('location_id') != process_location_id:
                continue
            if record.get('rack_type') != 'P':
                return result

            current = self.process_bottleneck_current.get(process_location_id, 0)
            if current > 0:
                current -= 1
                self.process_bottleneck_current[process_location_id] = current
                record['occupied'] = 1
                self.reserved_to_location_ids.discard(process_location_id)
                total = self.process_bottleneck_total.get(process_location_id, 0)
                result['work_messages'].append(
                    self._bottleneck_work_info(
                        message,
                        record,
                        '공정데포',
                        '대기품_작업시작',
                        current,
                        total,
                        delta_current=-1,
                        delta_total=0,
                    )
                )
                self._debug(
                    "process bottleneck moved to work: "
                    f"{process_location_id}, current={current}, total={total}"
                )
                return result

            if record.get('occupied') != 0:
                record['occupied'] = 0
                record['process_status'] = 'empty'
                self._debug(
                    "process depot released after worker completed: "
                    f"{process_location_id}"
                )

            self.reserved_to_location_ids.discard(process_location_id)
            result['release_messages'].append(
                self._process_release_message(message, record)
            )
            return result

        return result

    def _process_release_message(self, message, record):
        """P 공정데포 release 사실을 MCS queue retry 트리거로 보낸다."""
        payload = self._flat_location(record, message)
        payload['status'] = 'empty'
        return message.update(
            values={
                'info_type': 'process_released',
                'product': payload.get('product'),
                'part': payload.get('part'),
                'from': None,
                'to': payload,
                'location_id': payload.get('location_id'),
                'location_code': payload.get('location_code'),
                'node': payload.get('node'),
                'x': payload.get('x'),
                'y': payload.get('y'),
                'type': payload.get('type'),
                'capacity': payload.get('capacity'),
                'status': 'empty',
                'event_description': 'Process depot released',
            }
        )

    def _occupy_recovery_location(self, message):
        """Worker가 선택한 회수데포를 MM map에서도 occupied로 반영한다."""
        recovery_location_id = self._location_id(self._value(message, 'from'))
        if recovery_location_id is None:
            return

        for record in self.map_data.get('ml_nodes', []):
            if record.get('location_id') != recovery_location_id:
                continue
            if record.get('rack_type') != 'R':
                return
            if record.get('occupied') != 1:
                record['occupied'] = 1
                self._debug(
                    "recovery depot occupied after worker completed: "
                    f"{recovery_location_id}"
                )
            return

    def _process_location_id_from_worker_request(self, message):
        """Worker 요청에 남아 있는 원래 ML P 공정데포 위치를 찾는다."""
        for key in ('current_location', 'to', 'goal_location'):
            location_id = self._location_id(self._value(message, key))
            if location_id is None:
                continue

            for record in self.map_data.get('ml_nodes', []):
                if record.get('location_id') == location_id and record.get('rack_type') == 'P':
                    return location_id

        return None

    def _work_info_messages(self, message):
        """Analyzer에 보낼 AMR 작업 위치/상태 정보를 만든다."""
        depot_type = self._arrival_depot_type(message)
        if depot_type is None:
            return [self._work_info_payload(message, {}, None)]

        result_status = self._value(message, 'result_status')
        if result_status == 'LOAD_COMPLETED':
            status = 'Load'
            occupied = 0
        elif result_status == 'UNLOAD_COMPLETED':
            status = 'UnLoad'
            occupied = 1
        else:
            status = 'Load' if depot_type == '실대차 자재데포' else 'UnLoad'
            occupied = 0 if depot_type == '실대차 자재데포' else 1

        records = self._arrival_records(message, depot_type)
        if not records:
            return [self._work_info_payload(message, {'type': depot_type}, status)]

        for record in records:
            record['occupied'] = occupied
            location_id = record.get('location_id')
            if self._is_material_depot(depot_type):
                self.reserved_from_location_ids.discard(location_id)
            else:
                self.reserved_to_location_ids.discard(location_id)

        return [
            self._work_info_payload(message, record, status)
            for record in records
        ]

    def _recovery_update_messages(self, message):
        """회수데포 상차/하차 결과를 Worker 내부 R 상태로 동기화한다."""
        if self._arrival_depot_type(message) != '회수데포':
            return []

        result_status = self._value(message, 'result_status')
        if result_status == 'LOAD_COMPLETED':
            status = 'empty'
        elif result_status == 'UNLOAD_COMPLETED':
            status = 'occupied'
        else:
            return []

        messages = []
        for record in self._arrival_records(message, '회수데포'):
            payload = self._flat_location(record, message)
            payload['status'] = status
            messages.append(
                message.update(
                    values={
                        **payload,
                        'info_type': 'recovery_update',
                        'from': None,
                        'to': None,
                        'event_description': 'Recovery depot status updated',
                    }
                )
            )
        return messages

    def _process_arrival_messages(self, message, occupied_before=None):
        """공정데포 도착 시 Worker에 보낼 arrive_to_wk 메시지를 만든다."""
        result = {
            'signal_message': None,
            'mcs_messages': [],
            'work_messages': [],
        }
        if not self._is_process_unload_completed(message):
            return result

        records = self._arrival_records(message, '공정데포')
        if not records:
            return result

        signals = []
        for record in records:
            location_id = record.get('location_id')
            slot = self._process_slot(record)
            previous_occupied = (occupied_before or {}).get(location_id, record.get('occupied'))

            if self._is_process_bottleneck(location_id, previous_occupied):
                current = self.process_bottleneck_current.get(location_id, 0) + 1
                total = self.process_bottleneck_total.get(location_id, 0) + 1
                self.process_bottleneck_current[location_id] = current
                self.process_bottleneck_total[location_id] = total
                self._debug(
                    "process bottleneck arrival: "
                    f"location={location_id}, current={current}, total={total}"
                )
                result['work_messages'].append(
                    self._bottleneck_work_info(
                        message,
                        record,
                        '공정데포',
                        self._process_bottleneck_event_type(record),
                        current,
                        total,
                        delta_current=1,
                        delta_total=1,
                    )
                )

            record['occupied'] = 1
            ready_time = self.getTime() + self._part_work_time(message)
            record['process_status'] = 'processing'
            record['process_completed_time'] = ready_time
            self.process_state_by_location[location_id] = {
                'status': 'processing',
                'ready_time': ready_time,
                'record': deepcopy(record),
            }
            self._remember_supplied_process(record)
            self.reserved_to_location_ids.discard(location_id)

            if slot in (1, 2):
                signals.append(
                    message.update(
                        values=self._process_arrival_values(
                            record,
                            message,
                            f'Process depot P{slot} arrived',
                        )
                    )
                )
                continue

            self._debug(f"ignored process arrival with unsupported slot: {location_id}")

        if signals:
            result['signal_message'] = signals
        return result

    def _is_process_unload_completed(self, message):
        """실대차가 공정데포에 최종 UNLOAD 완료한 경우만 Worker 신호로 인정한다."""
        if self._arrival_depot_type(message) != '공정데포':
            return False
        if self._value(message, 'task_step') != 'from_to':
            return False
        if self._value(message, 'result_status') != 'UNLOAD_COMPLETED':
            return False
        if self._value(message, 'load_type') == '공대차':
            return False
        return True

    def _select_from_location(self, message):
        """MCS 요청에 맞는 WH 실대차 출발 위치를 하나 고른다."""
        records = self._records(
            'wh_locations',
            message,
            rack_type='O',
            occupied=1,
            use_location_id=True,
            excluded_location_ids=self.reserved_from_location_ids,
        )
        if not records:
            records = self._records(
                'wh_locations',
                message,
                rack_type='O',
                occupied=1,
                excluded_location_ids=self.reserved_from_location_ids,
            )
        return self._first_record(records)

    def _select_to_location(self, message):
        """target_slot 또는 제품/파트별 교대 슬롯을 우선 선택한다."""
        requested_slot = self._value(message, 'target_slot')
        if requested_slot is None and self._is_chain_supply(message):
            requested_slot = self.next_process_slot_by_key.get(
                self._product_part_key(message),
                1,
            )

        if requested_slot is not None:
            record = self._process_slot_record(
                message,
                requested_slot,
                occupied=0,
                excluded_location_ids=self.reserved_to_location_ids,
            )
            if record is not None:
                self._debug(
                    "selected process depot by requested slot: "
                    f"{record.get('location_id')}"
                )
                return record

            record = self._process_slot_record(
                message,
                requested_slot,
                excluded_location_ids=self.reserved_to_location_ids,
            )
            if record is not None:
                self._debug(
                    "selected occupied process depot for requested slot: "
                    f"{record.get('location_id')}"
                )
                return record

        for slot in (1, 2):
            record = self._process_slot_record(
                message,
                slot,
                occupied=0,
                excluded_location_ids=self.reserved_to_location_ids,
            )
            if record is not None:
                self._debug(
                    "selected process depot by P1/P2 fallback: "
                    f"{record.get('location_id')}"
                )
                return record

        records = self._records(
            'ml_nodes',
            message,
            rack_type='P',
        )
        if records:
            record = sorted(
                records,
                key=lambda candidate: (
                    self.process_bottleneck_current.get(candidate.get('location_id'), 0),
                    self._process_slot(candidate) or 999999,
                    str(candidate.get('location_id') or ''),
                ),
            )[0]
            self._debug(
                "selected occupied process depot for bottleneck: "
                f"{record.get('location_id')}, "
                f"current={self.process_bottleneck_current.get(record.get('location_id'), 0)}"
            )
            return record
        return None

    def _select_empty_cart_location(self, message):
        """Worker 요청에 맞는 WH rack_type X 위치를 하나 고른다."""
        empty_records = self._records(
            'wh_locations',
            message,
            rack_type='X',
            occupied=0,
            copy_record=False,
        )
        if empty_records:
            return self._first_record(empty_records)

        records = sorted(
            self._records(
                'wh_locations',
                message,
                rack_type='X',
                copy_record=False,
            ),
            key=self._record_sort_key,
        )
        if not records:
            return None

        key = (
            self._value(message, 'product'),
            self._value(message, 'part'),
        )
        index = self.empty_cart_fallback_indexes.get(key, 0) % len(records)
        self.empty_cart_fallback_indexes[key] = index + 1
        return records[index]

    def _empty_cart_location_payload(self, record, message):
        payload = self._flat_location(record, message)
        payload['empty_cart_bottleneck'] = bool(record.get('occupied') != 0)
        return payload

    def _arrival_records(self, message, depot_type):
        """AMR 도착 메시지에 해당하는 WH/ML map record를 찾는다."""
        location_id = self._message_location_id(message)
        if location_id is None:
            return []

        if self._is_material_depot(depot_type):
            rack_type = 'X' if depot_type == '공대차 자재데포' else 'O'
            return self._records(
                'wh_locations',
                message,
                rack_type=rack_type,
                location_id=location_id,
                copy_record=False,
            )
        rack_type = 'R' if depot_type == '회수데포' else 'P'
        return self._records(
            'ml_nodes',
            message,
            rack_type=rack_type,
            location_id=location_id,
            copy_record=False,
        )

    def _arrival_occupied_by_location(self, message, depot_type):
        """도착 처리 전에 해당 위치의 occupied 상태를 저장한다."""
        return {
            record.get('location_id'): record.get('occupied')
            for record in self._arrival_records(message, depot_type)
        }

    def _process_slot_record(
        self,
        message,
        slot,
        occupied=None,
        excluded_location_ids=None,
        copy_record=True,
    ):
        """같은 product/part의 P1 또는 P2 record를 찾는다."""
        try:
            slot = int(slot)
        except (TypeError, ValueError):
            return None

        records = self._records(
            'ml_nodes',
            message,
            rack_type='P',
            occupied=occupied,
            excluded_location_ids=excluded_location_ids,
            copy_record=copy_record,
        )
        for record in sorted(records, key=self._record_sort_key):
            if self._process_slot(record) == slot:
                return record
        return None

    def _process_slot(self, record):
        """P 공정데포 record에서 slot 번호를 읽는다."""
        try:
            return int(record.get('slot'))
        except (TypeError, ValueError):
            pass

        location_id = record.get('location_id')
        if not isinstance(location_id, str):
            return None
        suffix = location_id.rsplit('_P', 1)
        if len(suffix) != 2:
            return None
        try:
            return int(suffix[1])
        except ValueError:
            return None

    def _process_arrival_values(self, record, message, description):
        """Worker가 P1 작업 위치를 유지할 수 있게 위치 payload를 중복 세팅한다."""
        payload = self._flat_location(record, message)
        return {
            **payload,
            'current_location': payload,
            'to': payload,
            'target_slot': self._process_slot(record),
            'target_label': self._value(message, 'target_label'),
            'event_description': description,
        }

    def _is_process_bottleneck(self, location_id, previous_occupied):
        """해당 P가 이미 작업 중이거나 병목 대기열이 있으면 True."""
        return (
            previous_occupied != 0
            or self.process_bottleneck_current.get(location_id, 0) > 0
        )

    def _process_bottleneck_event_type(self, record):
        status = record.get('process_status')
        if status == 'empty_cart_ready':
            return '회수대기_진입'
        return '작업중_진입'

    def _empty_cart_return_has_bottleneck(self, message):
        for key in ('current_location', 'to', 'recovery_to'):
            location = self._value(message, key)
            if isinstance(location, dict) and location.get('empty_cart_bottleneck'):
                return True
        return False

    def _empty_cart_bottleneck_messages(self, message):
        if self._arrival_depot_type(message) != '공대차 자재데포':
            return []
        if self._value(message, 'task_step') != 'recovery_to_empty':
            return []
        if self._value(message, 'result_status') != 'UNLOAD_COMPLETED':
            return []
        if not self._empty_cart_return_has_bottleneck(message):
            return []

        work_messages = []
        for record in self._arrival_records(message, '공대차 자재데포'):
            location_id = record.get('location_id')
            current = self.empty_cart_bottleneck_current.get(location_id, 0) + 1
            total = self.empty_cart_bottleneck_total.get(location_id, 0) + 1
            self.empty_cart_bottleneck_current[location_id] = current
            self.empty_cart_bottleneck_total[location_id] = total
            self._debug(
                "empty cart depot bottleneck arrival: "
                f"location={location_id}, current={current}, total={total}"
            )
            work_messages.append(
                self._bottleneck_work_info(
                    message,
                    record,
                    '공대차 자재데포',
                    '공대차_반납병목_진입',
                    current,
                    total,
                    delta_current=1,
                    delta_total=1,
                )
            )
        return work_messages

    def _part_work_time(self, message):
        key = self._product_part_key(message)
        if key in PART_WORK_TIME:
            return PART_WORK_TIME[key]
        return PART_WORK_TIME.get(self._value(message, 'part'), 0)

    def _product_part_key(self, message_or_record):
        if isinstance(message_or_record, dict):
            return (
                message_or_record.get('product'),
                message_or_record.get('part'),
            )
        return (
            self._value(message_or_record, 'product'),
            self._value(message_or_record, 'part'),
        )

    def _is_chain_supply(self, message):
        if self._value(message, 'chain_recovery') is False:
            return False
        return True

    def _remember_supplied_process(self, record):
        key = self._product_part_key(record)
        slot = self._process_slot(record)
        self.last_supplied_process_by_key[key] = deepcopy(record)
        if slot in (1, 2):
            self.next_process_slot_by_key[key] = 2 if slot == 1 else 1

    def _chain_recovery_values(self, message, to_location):
        if not self._is_chain_supply(message) or to_location is None:
            return {
                'chain_recovery': False,
                'recovery_from': None,
                'recovery_to': None,
                'recovery_ready_time': None,
            }

        previous = self.last_supplied_process_by_key.get(self._product_part_key(message))
        if previous is None:
            return {
                'chain_recovery': False,
                'recovery_from': None,
                'recovery_to': None,
                'recovery_ready_time': None,
            }
        previous_location_id = previous.get('location_id')
        state = self.process_state_by_location.get(previous.get('location_id'), {})
        if (
            previous_location_id == to_location.get('location_id')
            and not self._has_recoverable_process_cart(previous_location_id, state)
        ):
            return {
                'chain_recovery': False,
                'recovery_from': None,
                'recovery_to': None,
                'recovery_ready_time': None,
            }
        if previous_location_id in self.reserved_process_recovery_location_ids:
            return {
                'chain_recovery': False,
                'recovery_from': None,
                'recovery_to': None,
                'recovery_ready_time': None,
            }

        if (
            state.get('status') == 'empty'
            and self.process_bottleneck_current.get(previous_location_id, 0) <= 0
        ):
            return {
                'chain_recovery': False,
                'recovery_from': None,
                'recovery_to': None,
                'recovery_ready_time': None,
            }

        recovery_to_record = self._select_empty_cart_location(message)
        if recovery_to_record is None:
            return {
                'chain_recovery': False,
                'recovery_from': None,
                'recovery_to': None,
                'recovery_ready_time': None,
            }
        recovery_to = self._empty_cart_location_payload(recovery_to_record, message)
        recovery_to_record['occupied'] = 1
        if previous_location_id is not None:
            self.reserved_process_recovery_location_ids.add(previous_location_id)

        recovery_from = self._flat_location(previous, message)
        recovery_from['process_status'] = state.get('status')
        recovery_from['process_completed_time'] = state.get('ready_time')

        return {
            'chain_recovery': True,
            'recovery_from': recovery_from,
            'recovery_to': recovery_to,
            'recovery_ready_time': state.get('ready_time') or self.getTime(),
        }

    def _has_recoverable_process_cart(self, location_id, state):
        if location_id is None:
            return False
        return (
            state.get('status') in ('processing', 'empty_cart_ready')
            or self.process_bottleneck_current.get(location_id, 0) > 0
        )

    def _mark_process_completed(self, message):
        location_id = self._location_id(
            self._value(message, 'from')
            or self._value(message, 'current_location')
            or self._value(message, 'to')
        )
        if location_id is None:
            return

        for record in self.map_data.get('ml_nodes', []):
            if record.get('location_id') != location_id:
                continue
            if record.get('rack_type') != 'P':
                return
            record['occupied'] = 1
            record['process_status'] = 'empty_cart_ready'
            record['process_completed_time'] = self._value(message, 'process_completed_time') or self.getTime()
            self.process_state_by_location[location_id] = {
                'status': 'empty_cart_ready',
                'ready_time': record['process_completed_time'],
                'record': deepcopy(record),
            }
            self._debug(f"process completed and waits for recovery: {location_id}")
            return

    def _handle_process_recovery_pickup(self, message):
        result = {
            'work_messages': [],
            'mcs_messages': [],
        }
        if self._arrival_depot_type(message) != '공정데포':
            return result
        if self._value(message, 'result_status') != 'LOAD_COMPLETED':
            return result
        if self._value(message, 'load_type') != '공대차':
            return result

        for record in self._arrival_records(message, '공정데포'):
            location_id = record.get('location_id')
            self.reserved_process_recovery_location_ids.discard(location_id)
            current = self.process_bottleneck_current.get(location_id, 0)
            total = self.process_bottleneck_total.get(location_id, 0)
            if current > 0:
                current -= 1
                self.process_bottleneck_current[location_id] = current
                result['work_messages'].append(
                    self._bottleneck_work_info(
                        message,
                        record,
                        '공정데포',
                        '공대차_회수완료',
                        current,
                        total,
                        delta_current=-1,
                        delta_total=0,
                    )
                )

            self.reserved_to_location_ids.discard(location_id)
            if current > 0:
                # 병목 대기 대차가 남아있으면 슬롯은 계속 점유 상태로 유지한다.
                # (worker 완료 경로 _release_process_location 과 동일한 규칙)
                record['occupied'] = 1
                follow_up = self._process_recovery_response(message, record)
                if follow_up is not None:
                    result['mcs_messages'].append(follow_up)
                self._debug(
                    "process empty cart picked up but bottleneck remains: "
                    f"{location_id}, current={current}"
                )
            else:
                record['occupied'] = 0
                record['process_status'] = 'empty'
                record['process_completed_time'] = None
                self.process_state_by_location[location_id] = {
                    'status': 'empty',
                    'ready_time': None,
                    'record': deepcopy(record),
                }
                self._debug(f"process empty cart picked up: {location_id}")
        return result

    def _process_recovery_response(self, message, process_record):
        """남은 공정데포 병목을 바로 빼기 위한 P -> X 회수 스케줄을 만든다."""
        location_id = process_record.get('location_id')
        if location_id in self.reserved_process_recovery_location_ids:
            return None

        empty_cart_record = self._select_empty_cart_location(message)
        if empty_cart_record is None:
            return None

        recovery_to = self._empty_cart_location_payload(empty_cart_record, message)
        empty_cart_record['occupied'] = 1
        self.reserved_process_recovery_location_ids.add(location_id)
        process_payload = self._flat_location(process_record, message)
        process_payload['status'] = 'empty_cart_ready'
        process_payload['process_status'] = 'empty_cart_ready'
        task_id = f"{location_id}_REC_{int(round(self.getTime())):06d}"

        return message.update(
            values={
                'info_type': 'response',
                'from': process_payload,
                'to': recovery_to,
                'task_id': task_id,
                'task_type': 'recovery',
                'task_step': None,
                'current_location': process_payload,
                'initial_leg': 'current_to_recovery',
                'chain_recovery': False,
                'recovery_from': process_payload,
                'recovery_to': recovery_to,
                'recovery_ready_time': None,
                'result_status': None,
                'load_type': None,
                'event_description': 'Process bottleneck recovery requested',
            }
        )

    def _request_from_process_location(self, message):
        location_id = self._location_id(self._value(message, 'from'))
        if location_id is None:
            return False
        for record in self.map_data.get('ml_nodes', []):
            if record.get('location_id') == location_id and record.get('rack_type') == 'P':
                return True
        return False

    def _bottleneck_work_info_from_message(self, message):
        """Worker가 보낸 R 병목 Message를 Analyzer용 dict로 변환한다."""
        location = self._value(message, 'from')
        record = location if isinstance(location, dict) else {}
        return self._bottleneck_work_info(
            message,
            record,
            self._value(message, 'bottleneck_depot_type') or record.get('type') or '미상',
            self._value(message, 'bottleneck_event_type') or '미완료_진입',
            self._value(message, 'bottleneck_current') or 0,
            self._value(message, 'bottleneck_total') or 0,
            self._value(message, 'bottleneck_delta_current') or 0,
            self._value(message, 'bottleneck_delta_total') or 0,
        )

    def _bottleneck_work_info(
        self,
        message,
        record,
        depot_type,
        event_type,
        current,
        total,
        delta_current,
        delta_total,
    ):
        """미완료 진입/대기 시작 이벤트를 Analyzer와 visualizer용으로 만든다."""
        payload = self._flat_location(record, message)
        timestamp = self._value(message, 'timestamp')
        if timestamp is None:
            timestamp = self.getTime()
        return {
            'AMR_id': self._value(message, 'AMR_id'),
            'AGV_id': self._value(message, 'AMR_id'),
            'task_id': self._value(message, 'task_id'),
            'task_type': self._value(message, 'task_type'),
            'task_step': self._value(message, 'task_step'),
            'move_num': self._value(message, 'move_num'),
            'generated_time': self._value(message, 'generated_time'),
            'cart_count': self._value(message, 'cart_count', 'cart'),
            'from': self._value(message, 'from'),
            'to': self._value(message, 'to'),
            'location_id': payload.get('location_id') or self._value(message, 'location_id'),
            'location_code': payload.get('location_code') or self._value(message, 'location_code'),
            'product': payload.get('product') or self._value(message, 'product'),
            'part': payload.get('part') or self._value(message, 'part'),
            'node': payload.get('node') or self._value(message, 'node'),
            'x': payload.get('x') or self._value(message, 'x'),
            'y': payload.get('y') or self._value(message, 'y'),
            'access_node_id': payload.get('access_node_id') or self._value(message, 'access_node_id'),
            'type': depot_type,
            'capacity': payload.get('capacity') or self._value(message, 'capacity'),
            'status': '병목',
            'load_type': self._value(message, 'load_type'),
            'timestamp': timestamp,
            'result_status': self._value(message, 'result_status'),
            'bottleneck_event': True,
            'bottleneck_event_type': event_type,
            'bottleneck_depot_type': depot_type,
            'bottleneck_current': int(current),
            'bottleneck_total': int(total),
            'bottleneck_delta_current': int(delta_current),
            'bottleneck_delta_total': int(delta_total),
        }

    def _records(
        self,
        collection_name,
        message,
        rack_type=None,
        occupied=None,
        location_id=None,
        use_location_id=False,
        excluded_location_ids=None,
        copy_record=True,
    ):
        """map.json collection에서 product/part/rack/occupied 조건에 맞는 record를 찾는다."""
        product = self._value(message, 'product')
        part = self._value(message, 'part')
        requested_id = location_id or self._value(message, 'location_id')
        excluded_location_ids = excluded_location_ids or set()

        records = []
        for record in self.map_data.get(collection_name, []):
            if record.get('location_id') in excluded_location_ids:
                continue
            if product is not None and record.get('product') != product:
                continue
            if part is not None and record.get('part') != part:
                continue
            if rack_type is not None and record.get('rack_type') != rack_type:
                continue
            if occupied is not None and record.get('occupied') != occupied:
                continue
            if use_location_id and requested_id is not None and record.get('location_id') != requested_id:
                continue
            if location_id is not None and record.get('location_id') != location_id:
                continue
            records.append(deepcopy(record) if copy_record else record)
        return records

    def _first_record(self, records):
        """후보 record 중 section/slot 순서로 하나를 고른다."""
        if not records:
            return None
        return sorted(records, key=self._record_sort_key)[0]

    def _record_sort_key(self, record):
        """WH section과 slot 값을 안정적으로 정렬한다."""
        section_order = {'F': 0, 'B': 1}
        try:
            slot = int(record.get('slot'))
        except (TypeError, ValueError):
            slot = 999999
        return (
            section_order.get(record.get('section'), 2),
            slot,
            str(record.get('location_id') or ''),
        )

    def _positive_int(self, value):
        """cart_count 같은 입력값을 0 이상의 정수로 변환한다."""
        if value is None:
            return None
        try:
            result = int(float(value))
        except (TypeError, ValueError):
            return None
        if result < 0:
            return 0
        return result

    def _arrival_depot_type(self, message):
        """AMR 완료 위치의 map record로 자재/공정 데포 도착을 판단한다."""
        record = self._arrival_record(message)
        if record is not None:
            return self._depot_type_from_record(record)

        text = ''.join(
            str(self._value(message, field) or '')
            for field in (
                'current_location',
                'goal_location',
                'to',
                'from',
                'arrival_event',
                'current_situation',
                'task_step',
                'type',
            )
        ).lower().replace(' ', '').replace('-', '_')

        if any(keyword in text for keyword in ('process_depot', 'production_depot', '공정데포', '공정대포')):
            return '공정데포'
        if any(keyword in text for keyword in ('empty_cart_depot', '공대차자재데포', '공대차')):
            return '공대차 자재데포'
        if any(keyword in text for keyword in ('material_depot', '자재데포', '자재대포')):
            return '실대차 자재데포'
        return None

    def _arrival_record(self, message):
        """AMR 완료 메시지의 location_id에 해당하는 map record를 찾는다."""
        location_id = self._message_location_id(message)
        if location_id is None:
            return None

        for collection_name in ('wh_locations', 'ml_nodes'):
            for record in self.map_data.get(collection_name, []):
                if record.get('location_id') == location_id:
                    return record
        return None

    def _depot_type_from_record(self, record):
        """map record의 zone/rack_type으로 데포 종류를 구분한다."""
        zone = record.get('zone')
        rack_type = record.get('rack_type')

        if zone == 'WH':
            if rack_type == 'O':
                return '실대차 자재데포'
            if rack_type == 'X':
                return '공대차 자재데포'
            return '자재데포'

        if zone == 'ML':
            if rack_type == 'P':
                return '공정데포'
            if rack_type == 'R':
                return '회수데포'

        return record.get('type')

    def _is_material_depot(self, depot_type):
        return depot_type in ('실대차 자재데포', '공대차 자재데포', '자재데포')

    def _message_location_id(self, message):
        """Message 필드나 from/to dict에서 location_id를 읽는다."""
        direct_id = self._value(message, 'location_id')
        if self._known_location_id(direct_id):
            return direct_id

        location_fields = ('to', 'goal_location', 'current_location', 'from')
        if self._value(message, 'result_status') is not None:
            location_fields = ('current_location', 'to', 'goal_location', 'from')

        for key in location_fields:
            location_id = self._location_id(self._value(message, key))
            if location_id is not None:
                return location_id
        return None

    def _location_id(self, value):
        """dict/list/string 위치값에서 location_id를 추출한다."""
        if isinstance(value, dict):
            location_id = value.get('location_id')
            return location_id if self._known_location_id(location_id) else None
        if isinstance(value, list):
            for entry in value:
                location_id = self._location_id(entry)
                if location_id is not None:
                    return location_id
            return None
        if self._known_location_id(value):
            return value
        return None

    def _known_location_id(self, value):
        """map.json에 실제 존재하는 location_id인지 확인한다."""
        if not isinstance(value, str):
            return False
        for collection_name in ('wh_locations', 'ml_nodes'):
            for record in self.map_data.get(collection_name, []):
                if record.get('location_id') == value:
                    return True
        return False

    def _flat_location(self, record, message):
        """map record를 MM 출력용 flat 위치 payload로 변환한다."""
        coordinates = record.get('coordinates') or {}
        location_type = record.get('type') or self._depot_type_from_record(record)

        payload = {
            'location_id': record.get('location_id'),
            'location_code': record.get('location_id'),
            'product': record.get('product'),
            'part': record.get('part'),
            'node': record.get('slot'),
            'x': coordinates.get('x', record.get('x')),
            'y': coordinates.get('y', record.get('y')),
            'access_node_id': record.get('access_node_id'),
            'type': location_type,
            'capacity': record.get('capacity', 1),
            'status': 'occupied' if record.get('occupied') else 'empty',
        }
        capacity = self._value(message, 'capacity')
        if capacity is not None:
            payload['capacity'] = capacity
        return payload

    def _work_info_payload(self, message, record, status):
        """work_info_to_analyzer에 필요한 dict를 만든다."""
        payload = self._flat_location(record, message)
        return {
            'AMR_id': self._value(message, 'AMR_id'),
            'AGV_id': self._value(message, 'AMR_id'),
            'task_id': self._value(message, 'task_id'),
            'task_type': self._value(message, 'task_type'),
            'task_step': self._value(message, 'task_step'),
            'move_num': self._value(message, 'move_num'),
            'generated_time': self._value(message, 'generated_time'),
            'cart_count': self._value(message, 'cart_count', 'cart'),
            'from': self._value(message, 'from'),
            'to': self._value(message, 'to'),
            'location_id': payload.get('location_id') or self._value(message, 'location_id'),
            'location_code': payload.get('location_code') or self._value(message, 'location_code'),
            'product': payload.get('product') or self._value(message, 'product'),
            'part': payload.get('part') or self._value(message, 'part'),
            'node': payload.get('node') or self._value(message, 'node'),
            'x': payload.get('x') or self._value(message, 'x'),
            'y': payload.get('y') or self._value(message, 'y'),
            'access_node_id': payload.get('access_node_id') or self._value(message, 'access_node_id'),
            'type': payload.get('type'),
            'capacity': payload.get('capacity'),
            'status': status if status is not None else payload.get('status'),
            'load_type': self._value(message, 'load_type'),
            'recovery_wait_time': self._value(message, 'recovery_wait_time'),
            'timestamp': self._value(message, 'timestamp'),
            'result_status': self._value(message, 'result_status'),
        }

    def _value(self, message, *keys):
        """Message에서 별칭을 포함해 첫 번째 non-None 값을 읽는다."""
        aliases = {
            'from': ('from_',),
            'part': ('part',),
            'AMR_id': ('AMR_id',),
            'amr_id': ('AMR_id',),
            'AGV_id': ('AMR_id',),
        }
        for key in keys:
            for attr in aliases.get(key, (key,)):
                if hasattr(message, attr):
                    value = getattr(message, attr)
                    if value is not None:
                        return value
        return None

    def _debug(self, text):
        """ManufacturingManager 디버깅 로그를 출력한다."""
        print(f"[{self.getTime():08.2f}][ManufacturingManage][DEBUG] {text}")

    def _describe_message(self, message):
        """디버깅용으로 Message 핵심 필드를 문자열로 만든다."""
        return (
            f"product={self._value(message, 'product')}, "
            f"part={self._value(message, 'part')}, "
            f"cart_count={self._value(message, 'cart_count', 'cart')}, "
            f"location_id={self._message_location_id(message)}, "
            f"info_type={self._value(message, 'info_type')}"
        )

    def _describe_event(self, event):
        """리스트/Message/dict 이벤트를 디버깅 문자열로 만든다."""
        if isinstance(event, list):
            ids = [
                self._location_id(entry.to_dict() if isinstance(entry, Message) else entry)
                for entry in event
            ]
            return f"count={len(event)}, locations={ids}"
        if isinstance(event, Message):
            return self._describe_message(event)
        if isinstance(event, dict):
            return f"location={self._location_id(event)}"
        return str(event)
