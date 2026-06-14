from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message

INF = float('inf')


class MCS(DEVSAtomicModel):

    def __init__(self, ID):
        super().__init__(ID)

        self.addStateVariable('state', 'WAIT')

        # 수신한 supply order를 임시 보관 (REQUEST 출력용)
        self._pending_order = None
        # m_info_from_mm에서 받은 스케줄 후보 (CHECK 출력용)
        self._pending_info = None
        # item_depot_update_from_wk에서 받은 update 정보 (SCHEDULE 출력용)
        self._pending_update = None
        # scheduling 불가 시 대기 queue: list of Message (product, part 포함)
        self._queue = []
        self._inflight_order = None
        self._pending_release_updates = []
        self._last_queue_retry_time = None
        self.input_queue = []

        self.addInputPort('supply_order_from_gen')
        self.addInputPort('m_info_from_mm')
        self.addInputPort('item_depot_update_from_wk')

        self.addOutputPort('m_req_to_mm')
        self.addOutputPort('schedule_info_to_ACS')

    def funcExternalTransition(self, strPort, event):
        state = self.getStateValue('state')
        self.input_queue.append((strPort, self._to_message(event)))
        if state == 'WAIT':
            self._start_next_job()
        else:
            self.continueTimeAdvance()

    def funcOutput(self):
        state = self.getStateValue('state')

        if state == 'REQUEST':
            # supply order generator에서 받은 message를 그대로 MM으로 전달
            print(
                f"[{self.getTime():08.2f}][MCS] REQUEST: "
                f"product={self._pending_order.product}, "
                f"part={self._pending_order.part}, "
                f"target_slot={self._pending_order.target_slot}, "
                f"cart_count={self._pending_order.cart_count}"
            )
            self._inflight_order = self._pending_order
            self.addOutputEvent('m_req_to_mm', self._pending_order)

        elif state == 'SCHEDULE':
            # item_depot_update_from_wk로부터 받은 update를 그대로 ACS으로 전달
            for update in self._pending_update.item_list or []:
                print(f"[{self.getTime():08.2f}][MCS] SCHEDULE: product={update.product}, part={update.part}, cart_count={update.cart_count}")
                self.addOutputEvent('schedule_info_to_ACS', update)

        elif state == 'CHECK':
            info = self._pending_info
            info_type = info.info_type  # "response" or "update"

            if info_type == 'response':
                if info.from_ is not None and info.to is not None:
                    # scheduling 가능: 주문 메타데이터를 유지한 채 ACS로 전달
                    msg = info.update(values={
                        'from': info.from_,
                        'to': info.to,
                        'product': self._message_product(info),
                        'part': self._message_part(info),
                        'target_slot': self._message_target_slot(info),
                    })
                    print(f"[{self.getTime():08.2f}][MCS] schedule info to ACS: {msg.from_['location_id']} -> {msg.to['location_id']}")
                    self.addOutputEvent('schedule_info_to_ACS', msg)
                    self._inflight_order = None
                else:
                    # scheduling 불가: queue에 적재 후 출력 없음
                    # _inflight_order에 원래 supply order가 남아 있으므로 그 정보를 저장
                    if self._inflight_order is not None:
                        self._queue.append(self._inflight_order)
                    self._inflight_order = None
                self._retry_pending_release_updates()

            elif info_type == 'update':
                # depot에 새 자재가 입고됨 → queue에서 product/part가 일치하는 항목 처리
                if info.from_ is None:
                    print(f"[{self.getTime():08.2f}][MCS] update ignored: from is None")
                    return

                matched = None
                for queued in self._queue:
                    if self._same_order_key(queued, info):
                        matched = queued
                        break
                if matched is not None:
                    if info.to is None:
                        print(f"[{self.getTime():08.2f}][MCS] update matched but to is None")
                        return

                    self._queue.remove(matched)
                    msg = matched.update(values={
                        'from': info.from_,
                        'to': info.to,
                        'product': self._message_product(info) or matched.product,
                        'part': self._message_part(info) or matched.part,
                        'target_slot': self._message_target_slot(info) or matched.target_slot,
                    })
                    print(f"[{self.getTime():08.2f}][MCS] schedule info to ACS: {msg.from_['location_id']} -> {msg.to['location_id']}")
                    self.addOutputEvent('schedule_info_to_ACS', msg)

            elif info_type == 'process_released':
                self._pending_release_updates.append(info)
                self._retry_pending_release_updates()

    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            pass
        elif state == 'REQUEST':
            self.setStateValue('state', 'WAIT')
            self._pending_order = None
            self._start_next_job()
        elif state == 'SCHEDULE':
            self.setStateValue('state', 'WAIT')
            self._pending_update = None
            self._start_next_job()
        elif state == 'CHECK':
            self.setStateValue('state', 'WAIT')
            self._pending_info = None
            self._start_next_job()

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            return INF
        elif state == 'REQUEST':
            return 0
        elif state == 'SCHEDULE':
            return 0
        elif state == 'CHECK':
            return 0

    def funcSelect(self):
        pass

    def _to_message(self, event):
        if isinstance(event, Message):
            return event.copy()
        return Message.from_dict(event)

    def _start_next_job(self):
        if self._should_retry_queued_order():
            self._pending_order = self._queue.pop(0)
            self._last_queue_retry_time = self.getTime()
            self.setStateValue('state', 'REQUEST')
            return

        while self.input_queue:
            input_index = self._next_processable_input_index()
            if input_index is None:
                break

            strPort, message = self.input_queue.pop(input_index)

            if strPort == 'supply_order_from_gen':
                self._pending_order = message
                self.setStateValue('state', 'REQUEST')
                return

            if strPort == 'm_info_from_mm':
                self._pending_info = message
                self.setStateValue('state', 'CHECK')
                return

            if strPort == 'item_depot_update_from_wk':
                self._pending_update = message
                self.setStateValue('state', 'SCHEDULE')
                return

        self.setStateValue('state', 'WAIT')

    def _next_processable_input_index(self):
        if self._inflight_order is None:
            return 0 if self.input_queue else None

        for index, (strPort, _message) in enumerate(self.input_queue):
            if strPort != 'supply_order_from_gen':
                return index
        return None

    def _retry_pending_release_updates(self):
        if self._inflight_order is not None:
            return

        for release in list(self._pending_release_updates):
            matched = self._pop_matching_queued_order(release)
            if matched is None:
                continue

            self._pending_release_updates.remove(release)
            self._inflight_order = matched
            self._last_queue_retry_time = self.getTime()
            print(
                f"[{self.getTime():08.2f}][MCS] RETRY: "
                f"product={matched.product}, part={matched.part}"
            )
            self.addOutputEvent('m_req_to_mm', matched)
            return

    def _should_retry_queued_order(self):
        return (
            self._inflight_order is None
            and len(self._queue) > 0
            and self._last_queue_retry_time != self.getTime()
        )

    def _pop_matching_queued_order(self, info):
        product = self._message_product(info)
        part = self._message_part(info)

        for queued in list(self._queue):
            if self._same_order_key(queued, info):
                self._queue.remove(queued)
                return queued
        return None

    def _message_product(self, message):
        if message.product is not None:
            return message.product
        for location in (message.from_, message.to):
            if isinstance(location, dict) and location.get('product') is not None:
                return location.get('product')
        return None

    def _message_part(self, message):
        if message.part is not None:
            return message.part
        for location in (message.from_, message.to):
            if isinstance(location, dict) and location.get('part') is not None:
                return location.get('part')
        return None

    def _message_target_slot(self, message):
        if message.target_slot is not None:
            return message.target_slot
        for location in (message.from_, message.to):
            if isinstance(location, dict) and location.get('node') is not None:
                return location.get('node')
        return None

    def _same_order_key(self, queued, info):
        product = self._message_product(info)
        part = self._message_part(info)
        target_slot = self._message_target_slot(info)
        if queued.product != product or queued.part != part:
            return False
        if queued.target_slot is None or target_slot is None:
            return True
        return str(queued.target_slot) == str(target_slot)
