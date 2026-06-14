import atexit
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from config import AMR_LOAD_UNLOAD_TIME

INF = float('inf')


class Analyzer(DEVSAtomicModel):

    def __init__(self, ID):
        super().__init__(ID)

        self.addStateVariable('state', 'GEN')

        self.addInputPort('operation_result_from_ACS')
        self.addInputPort('physical_result_from_amr')
        self.addInputPort('work_info_from_mm')

        self.simulation_end_time = 0.0
        self.assignment_count = 0
        self.material_to_process_count = 0
        self.empty_cart_return_count = 0
        self.recovery_pickup_count = 0

        self.orders = {}
        self.amr_stats = defaultdict(self._new_amr_stats)
        self.amr_work_start_times = {}
        self.task_step_stats = defaultdict(self._new_step_stats)
        self.depot_event_counts = Counter()
        self.bottleneck_by_location = {}
        self.bottleneck_events = []
        self.recovery_wait_times = []
        self.recovery_wait_times_by_part = defaultdict(list)
        self.recovery_wait_times_by_amr = defaultdict(list)

        self.output_dir = Path(__file__).resolve().parents[3] / 'simulation_log'
        self.output_path = None
        self._exported_once = False
        self._manual_export_done = False
        atexit.register(self._export_kpi_at_exit)

    def funcExternalTransition(self, strPort, event):
        state = self.getStateValue('state')
        if state == 'GEN' :
            if strPort == 'operation_result_from_ACS':
                self._record_operation_result(event)
            elif strPort == 'physical_result_from_amr':
                self._record_physical_result(event)
            elif strPort == 'work_info_from_mm':
                self._record_work_info(event)
            self.setStateValue('state', 'GEN')

    def funcOutput(self):
        pass

    def funcInternalTransition(self):
        pass

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'GEN':
            return INF
        return INF

    def funcSelect(self):
        pass

    def _record_operation_result(self, event):
        result_status = self._value(event, 'result_status')
        event_time = self._event_time(event)
        self._touch_end_time(event_time)

        if result_status != 'ASSIGNED':
            return

        self.assignment_count += 1
        amr_id = self._value(event, 'AMR_id')
        if amr_id is not None:
            self.amr_stats[amr_id]['배정횟수'] += 1
            self._start_amr_work_period(amr_id, event_time)

        if not self._is_material_to_process(event):
            return

        order_id = self._order_id(event)
        if order_id is None:
            return

        order = self._order_record(order_id, event)
        if order.get('배정시간') is None:
            order['배정시간'] = event_time
            order['대기시간'] = self._duration(order.get('주문발생시간'), event_time)
        order['출발자재데포'] = self._location_id(self._value(event, 'from'))
        order['도착공정데포'] = self._location_id(self._value(event, 'to'))

    def _record_physical_result(self, event):
        event_time = self._event_time(event)
        self._touch_end_time(event_time)

        if self._value(event, 'result_status') != 'SUCCESS':
            return

        amr_id = self._value(event, 'AMR_id')
        raw_task_step = self._value(event, 'task_step')
        task_step = self._translate_task_step(raw_task_step)
        travel_time = self._float_value(self._value(event, 'travel_time'))
        travel_distance = self._float_value(self._value(event, 'travel_distance'))

        if amr_id is not None:
            stats = self.amr_stats[amr_id]
            stats['이동완료횟수'] += 1
            stats['이동시간'] += travel_time
            stats['이동거리'] += travel_distance
            if raw_task_step in ('to_station', 'empty_to_station'):
                self._finish_amr_work_period(amr_id, event_time)

        step_stats = self.task_step_stats[task_step]
        step_stats['이동완료횟수'] += 1
        step_stats['이동시간'] += travel_time
        step_stats['이동거리'] += travel_distance

    def _record_work_info(self, event):
        for item in self._items(event):
            event_time = self._event_time(item)
            self._touch_end_time(event_time)

            if self._value(item, 'bottleneck_event'):
                self._record_bottleneck_event(item, event_time)
                continue

            depot_type = self._value(item, 'type') or '미상'
            status = self._translate_status(self._value(item, 'status'))
            self.depot_event_counts[(depot_type, status)] += 1
            self._record_amr_work_time(item)
            self._record_recovery_wait(item)

            if self._is_process_unload(item):
                self.material_to_process_count += 1
                self._record_supply_completed(item, event_time)
            elif self._is_empty_cart_return(item):
                self.empty_cart_return_count += 1
            elif self._is_recovery_pickup(item):
                self.recovery_pickup_count += 1

    def _record_supply_completed(self, item, event_time):
        order_id = self._order_id(item)
        if order_id is None:
            return

        order = self._order_record(order_id, item)
        if order.get('공급완료시간') is None:
            order['공급완료시간'] = event_time
            order['공급시간'] = self._duration(order.get('주문발생시간'), event_time)

    def _record_bottleneck_event(self, item, event_time):
        location_id = (
            self._value(item, 'location_id')
            or self._location_id(self._value(item, 'from'))
            or self._location_id(self._value(item, 'to'))
            or '미상'
        )
        depot_type = (
            self._value(item, 'bottleneck_depot_type')
            or self._value(item, 'type')
            or '미상'
        )
        event_type = self._value(item, 'bottleneck_event_type') or '미완료_진입'
        current = self._int_value(self._value(item, 'bottleneck_current'))
        total = self._int_value(self._value(item, 'bottleneck_total'))
        delta_current = self._int_value(self._value(item, 'bottleneck_delta_current')) or 0
        delta_total = self._int_value(self._value(item, 'bottleneck_delta_total')) or 0

        detail = self.bottleneck_by_location.setdefault(
            location_id,
            {
                '데포ID': location_id,
                '데포종류': depot_type,
                '제품': self._event_product(item),
                '부품': self._event_part(item),
                '현재_미완료_진입수': 0,
                '누적_미완료_진입수': 0,
            },
        )

        if detail.get('제품') is None:
            detail['제품'] = self._event_product(item)
        if detail.get('부품') is None:
            detail['부품'] = self._event_part(item)
        detail['데포종류'] = depot_type

        if current is None:
            current = max(0, detail['현재_미완료_진입수'] + delta_current)
        if total is None:
            total = max(0, detail['누적_미완료_진입수'] + delta_total)

        detail['현재_미완료_진입수'] = max(0, current)
        detail['누적_미완료_진입수'] = max(
            detail['누적_미완료_진입수'],
            total,
        )

        self.bottleneck_events.append({
            '시간': self._round(event_time),
            '데포ID': location_id,
            '데포종류': depot_type,
            '이벤트': event_type,
            '현재_미완료_진입수': detail['현재_미완료_진입수'],
            '누적_미완료_진입수': detail['누적_미완료_진입수'],
            '변화_현재': delta_current,
            '변화_누적': delta_total,
            'AMR_ID': self._value(item, 'AMR_id'),
            '작업ID': self._order_id(item),
            '제품': self._event_product(item),
            '부품': self._event_part(item),
        })

    def get_kpi(self):
        wait_times = [
            order['대기시간']
            for order in self.orders.values()
            if order.get('대기시간') is not None
        ]
        supply_times = [
            order['공급시간']
            for order in self.orders.values()
            if order.get('공급시간') is not None
        ]

        return {
            '지표_정의': {
                '총_자재데포_공정데포_이송횟수': '공정데포에 실대차 UnLoad가 완료된 횟수',
                '이동로봇별_가동률': '이동로봇별 이동시간 합계 / 시뮬레이션 종료시간',
                '이동로봇별_작업가동률': '이동로봇별 station 대기 시간을 제외한 작업가동시간 / 시뮬레이션 종료시간',
                '평균_대기시간': '주문 발생 시간부터 ACS 배정 시간까지의 평균',
                '평균_공급시간': '주문 발생 시간부터 공정데포 UnLoad 완료 시간까지의 평균',
                '파트별_자재데포_공급지표': '제품/파트가 같은 자재데포들을 묶은 완료 이송횟수와 평균 대기/공급시간',
                '데포별_병목지표': '작업/회수가 끝나지 않은 데포에 실제 하차 또는 도착 완료된 대차의 누적 수',
            },
            '시뮬레이션_종료시간': self._round(self.simulation_end_time),
            '총_작업배정횟수': self.assignment_count,
            '총_자재데포_공정데포_이송횟수': self.material_to_process_count,
            '총_공대차_자재데포_반납횟수': self.empty_cart_return_count,
            '총_회수데포_상차횟수': self.recovery_pickup_count,
            '총_회수_대기시간': self._round(sum(self.recovery_wait_times)),
            '평균_회수_대기시간': self._average(self.recovery_wait_times),
            '파트별_회수_대기시간': self._recovery_wait_by_part_summary(),
            'AMR별_회수_대기시간': self._recovery_wait_by_amr_summary(),
            '평균_대기시간': self._average(wait_times),
            '평균_공급시간': self._average(supply_times),
            '대기시간_집계주문수': len(wait_times),
            '공급시간_집계주문수': len(supply_times),
            '이동로봇별_가동률': self._amr_utilization(),
            '이동로봇별_작업가동률': self._amr_work_utilization(),
            '파트별_자재데포_공급지표': self._material_depot_summary(),
            '작업단계별_이동실적': self._step_summary(),
            '데포이벤트_건수': self._depot_event_summary(),
            '총_공정데포_병목횟수': self._bottleneck_total('공정데포'),
            '총_회수데포_병목횟수': self._bottleneck_total('회수데포'),
            '데포별_병목지표': self._bottleneck_summary(),
            '병목_이벤트': self._bottleneck_event_summary(),
            '주문별_지표': self._order_summary(),
        }

    def export_kpi(self, manual=True):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = self.get_kpi()

        if self.output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_path = self.output_dir / f'kpi_{timestamp}.json'

        self.output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        self._exported_once = True

        if manual:
            self._manual_export_done = True

        print(f"[Analyzer] KPI 저장 완료: {self.output_path}")

    def _export_kpi_at_exit(self):
        if self._manual_export_done:
            return
        self.export_kpi(manual=False)

    def _new_amr_stats(self):
        return {
            '배정횟수': 0,
            '이동완료횟수': 0,
            '이동시간': 0.0,
            '상하차시간': 0.0,
            '회수대기시간': 0.0,
            '병목대기시간': 0.0,
            '작업가동시간': 0.0,
            '이동거리': 0.0,
        }

    def _new_step_stats(self):
        return {
            '이동완료횟수': 0,
            '이동시간': 0.0,
            '이동거리': 0.0,
        }

    def _amr_utilization(self):
        summary = {}
        for amr_id in sorted(self.amr_stats):
            stats = self.amr_stats[amr_id]
            utilization = 0.0
            if self.simulation_end_time > 0:
                utilization = stats['이동시간'] / self.simulation_end_time

            summary[amr_id] = {
                '가동률': self._round(utilization),
                '가동률_퍼센트': self._round(utilization * 100),
                '배정횟수': stats['배정횟수'],
                '이동완료횟수': stats['이동완료횟수'],
                '이동시간': self._round(stats['이동시간']),
                '이동거리': self._round(stats['이동거리']),
            }
        return summary

    def _amr_work_utilization(self):
        summary = {}
        for amr_id in sorted(self.amr_stats):
            stats = self.amr_stats[amr_id]
            work_time = self._amr_work_time(amr_id)
            utilization = 0.0
            if self.simulation_end_time > 0:
                utilization = work_time / self.simulation_end_time

            summary[amr_id] = {
                '작업가동률': self._round(utilization),
                '작업가동률_퍼센트': self._round(utilization * 100),
                '작업가동시간': self._round(work_time),
                '이동시간': self._round(stats['이동시간']),
                '상하차시간': self._round(stats['상하차시간']),
                '회수대기시간': self._round(stats['회수대기시간']),
                '병목대기시간': self._round(stats['병목대기시간']),
            }
        return summary

    def _start_amr_work_period(self, amr_id, event_time):
        if amr_id not in self.amr_work_start_times:
            self.amr_work_start_times[amr_id] = event_time

    def _finish_amr_work_period(self, amr_id, event_time):
        start_time = self.amr_work_start_times.pop(amr_id, None)
        if start_time is None:
            return
        self.amr_stats[amr_id]['작업가동시간'] += max(0.0, event_time - start_time)

    def _amr_work_time(self, amr_id):
        work_time = self.amr_stats[amr_id]['작업가동시간']
        start_time = self.amr_work_start_times.get(amr_id)
        if start_time is not None:
            work_time += max(0.0, self.simulation_end_time - start_time)
        return work_time

    def _step_summary(self):
        return {
            step: {
                '이동완료횟수': stats['이동완료횟수'],
                '이동시간': self._round(stats['이동시간']),
                '이동거리': self._round(stats['이동거리']),
            }
            for step, stats in sorted(self.task_step_stats.items())
        }

    def _material_depot_summary(self):
        grouped = {}
        for order in self._completed_material_orders():
            product = order.get('제품') or '미상'
            part = order.get('부품') or '미상'
            group_key = f"{product}_{part}_자재데포"

            stats = grouped.setdefault(
                group_key,
                {
                    '제품': product,
                    '부품': part,
                    '자재데포그룹': group_key,
                    '이송횟수': 0,
                    '포함_자재데포': set(),
                    '대기시간목록': [],
                    '공급시간목록': [],
                },
            )
            stats['이송횟수'] += 1
            if order.get('출발자재데포') is not None:
                stats['포함_자재데포'].add(order['출발자재데포'])
            if order.get('대기시간') is not None:
                stats['대기시간목록'].append(order['대기시간'])
            if order.get('공급시간') is not None:
                stats['공급시간목록'].append(order['공급시간'])

        return {
            stats['자재데포그룹']: {
                '제품': stats['제품'],
                '부품': stats['부품'],
                '포함_자재데포': sorted(stats['포함_자재데포']),
                '이송횟수': stats['이송횟수'],
                '평균_대기시간': self._average(stats['대기시간목록']),
                '평균_공급시간': self._average(stats['공급시간목록']),
                '대기시간_집계주문수': len(stats['대기시간목록']),
                '공급시간_집계주문수': len(stats['공급시간목록']),
            }
            for _group_key, stats in sorted(grouped.items())
        }

    def _completed_material_orders(self):
        return [
            order
            for order in self.orders.values()
            if order.get('공급완료시간') is not None
            and order.get('출발자재데포') is not None
        ]

    def _depot_event_summary(self):
        summary = {}
        for (depot_type, status), count in sorted(self.depot_event_counts.items()):
            summary.setdefault(depot_type, {})[status] = count
        return summary

    def _record_recovery_wait(self, item):
        wait_time = self._float_value(self._value(item, 'recovery_wait_time'))
        if wait_time is None:
            return
        if wait_time < 0:
            wait_time = 0.0
        self.recovery_wait_times.append(wait_time)
        product = self._event_product(item) or '미상'
        part = self._event_part(item) or '미상'
        self.recovery_wait_times_by_part[(product, part)].append(wait_time)
        amr_id = self._value(item, 'AMR_id') or '미상'
        self.recovery_wait_times_by_amr[amr_id].append(wait_time)
        if amr_id != '미상' and self._value(item, 'status') == 'Load':
            self.amr_stats[amr_id]['회수대기시간'] += wait_time

    def _record_amr_work_time(self, item):
        if self._value(item, 'status') not in ('Load', 'UnLoad'):
            return
        amr_id = self._value(item, 'AMR_id')
        if amr_id is None:
            return
        load_unload_time = self._float_value(AMR_LOAD_UNLOAD_TIME)
        if load_unload_time is None:
            return
        self.amr_stats[amr_id]['상하차시간'] += load_unload_time

    def _recovery_wait_by_part_summary(self):
        return {
            f"{product}_{part}": {
                '제품': product,
                '부품': part,
                '총_회수_대기시간': self._round(sum(values)),
                '평균_회수_대기시간': self._average(values),
                '건수': len(values),
            }
            for (product, part), values in sorted(self.recovery_wait_times_by_part.items())
        }

    def _recovery_wait_by_amr_summary(self):
        return {
            amr_id: {
                '총_회수_대기시간': self._round(sum(values)),
                '평균_회수_대기시간': self._average(values),
                '건수': len(values),
            }
            for amr_id, values in sorted(self.recovery_wait_times_by_amr.items())
        }

    def _bottleneck_total(self, depot_type):
        return sum(
            detail.get('누적_미완료_진입수', 0)
            for detail in self.bottleneck_by_location.values()
            if detail.get('데포종류') == depot_type
        )

    def _bottleneck_summary(self):
        return {
            location_id: {
                '데포ID': detail.get('데포ID'),
                '데포종류': detail.get('데포종류'),
                '제품': detail.get('제품'),
                '부품': detail.get('부품'),
                '누적_병목횟수': detail.get('누적_미완료_진입수', 0),
            }
            for location_id, detail in sorted(self.bottleneck_by_location.items())
        }

    def _bottleneck_event_summary(self):
        return [
            {
                '시간': event.get('시간'),
                '데포ID': event.get('데포ID'),
                '데포종류': event.get('데포종류'),
                '이벤트': self._translate_bottleneck_event(event.get('이벤트')),
                '누적_병목횟수': event.get('누적_미완료_진입수'),
                '변화_누적': event.get('변화_누적'),
                'AMR_ID': event.get('AMR_ID'),
                '작업ID': event.get('작업ID'),
                '제품': event.get('제품'),
                '부품': event.get('부품'),
            }
            for event in self.bottleneck_events
        ]

    def _translate_bottleneck_event(self, event_type):
        labels = {
            '미완료_진입': '병목_진입',
            '작업중_진입': '작업중_진입',
            '회수대기_진입': '회수대기_진입',
            '대기품_작업시작': '병목_작업시작',
            '대기품_회수시작': '병목_회수시작',
            '공대차_회수완료': '병목_회수완료',
        }
        return labels.get(event_type, event_type or '병목')

    def _order_summary(self):
        return {
            order_id: {
                key: self._round(value) if isinstance(value, float) else value
                for key, value in order.items()
            }
            for order_id, order in sorted(self.orders.items())
        }

    def _order_record(self, order_id, event):
        if order_id not in self.orders:
            self.orders[order_id] = {
                '주문ID': order_id,
                '제품': self._event_product(event),
                '부품': self._event_part(event),
                '카트수량': self._event_cart_count(event),
                '주문발생시간': self._event_generated_time(event),
                '배정시간': None,
                '공급완료시간': None,
                '대기시간': None,
                '공급시간': None,
                '출발자재데포': self._location_id(self._value(event, 'from')),
                '도착공정데포': self._location_id(self._value(event, 'to')),
            }
        else:
            order = self.orders[order_id]
            if order.get('주문발생시간') is None:
                order['주문발생시간'] = self._event_generated_time(event)
            if order.get('제품') is None:
                order['제품'] = self._event_product(event)
            if order.get('부품') is None:
                order['부품'] = self._event_part(event)
            if order.get('카트수량') is None:
                order['카트수량'] = self._event_cart_count(event)
            if order.get('출발자재데포') is None:
                order['출발자재데포'] = self._location_id(self._value(event, 'from'))
            if order.get('도착공정데포') is None:
                order['도착공정데포'] = self._location_id(self._value(event, 'to'))
        return self.orders[order_id]

    def _is_material_to_process(self, event):
        from_location = self._value(event, 'from')
        to_location = self._value(event, 'to')
        return (
            self._location_zone(from_location) == 'WH'
            and self._rack_type(from_location) == 'O'
            and self._location_zone(to_location) == 'ML'
            and self._rack_type(to_location) == 'P'
        )

    def _is_process_unload(self, item):
        return (
            self._value(item, 'type') == '공정데포'
            and self._value(item, 'status') == 'UnLoad'
            and self._value(item, 'load_type') != '공대차'
        )

    def _is_empty_cart_return(self, item):
        return (
            self._value(item, 'type') == '공대차 자재데포'
            and self._value(item, 'status') == 'UnLoad'
        )

    def _is_recovery_pickup(self, item):
        return (
            self._value(item, 'type') == '회수데포'
            and self._value(item, 'status') == 'Load'
        )

    def _items(self, event):
        if isinstance(event, list):
            return event
        return [event]

    def _value(self, event, key):
        attr = 'from_' if key == 'from' else key
        if isinstance(event, dict):
            if key == 'from':
                return event.get('from', event.get('from_'))
            return event.get(key)
        return getattr(event, attr, None)

    def _order_id(self, event):
        order_id = self._value(event, 'order_id')
        if order_id is not None:
            return order_id
        task_id = self._value(event, 'task_id')
        if task_id is not None:
            return task_id
        move_num = self._value(event, 'move_num')
        if move_num is not None:
            return f"move_{move_num}"
        return None

    def _event_product(self, event):
        product = self._value(event, 'product')
        if product is not None:
            return product
        for key in ('from', 'to', 'current_location'):
            location = self._value(event, key)
            if isinstance(location, dict) and location.get('product') is not None:
                return location.get('product')
        return None

    def _event_part(self, event):
        part = self._value(event, 'part')
        if part is not None:
            return part
        for key in ('from', 'to', 'current_location'):
            location = self._value(event, key)
            if isinstance(location, dict) and location.get('part') is not None:
                return location.get('part')
        return None

    def _event_cart_count(self, event):
        cart_count = self._value(event, 'cart_count')
        if cart_count is not None:
            return cart_count
        return self._value(event, 'cart')

    def _event_generated_time(self, event):
        return self._float_value(self._value(event, 'generated_time'))

    def _event_time(self, event):
        timestamp = self._float_value(self._value(event, 'timestamp'))
        if timestamp is not None:
            return timestamp
        return self._float_value(self.getTime()) or 0.0

    def _duration(self, start, end):
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    def _float_value(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int_value(self, value):
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _touch_end_time(self, time_value):
        if time_value is not None:
            self.simulation_end_time = max(self.simulation_end_time, float(time_value))

    def _average(self, values):
        if not values:
            return None
        return self._round(sum(values) / len(values))

    def _round(self, value):
        if value is None:
            return None
        return round(float(value), 4)

    def _location_zone(self, location):
        if isinstance(location, dict):
            return location.get('zone')
        return None

    def _rack_type(self, location):
        if isinstance(location, dict):
            return location.get('rack_type')
        return None

    def _location_id(self, location):
        if isinstance(location, dict):
            return location.get('location_id') or location.get('location_code')
        if isinstance(location, str):
            return location
        return None

    def _translate_task_step(self, task_step):
        labels = {
            'current_to_from': '현재위치_출발지',
            'from_to': '출발지_도착지',
            'to_station': '도착지_스테이션',
            'to_recovery': '공급지_회수지',
            'current_to_recovery': '현재위치_회수지',
            'recovery_to_empty': '회수지_공대차데포',
            'empty_to_station': '공대차데포_스테이션',
        }
        return labels.get(task_step, task_step or '미상')

    def _translate_status(self, status):
        labels = {
            'Load': '상차',
            'UnLoad': '하차',
            'occupied': '점유',
            'empty': '비어있음',
        }
        return labels.get(status, status or '미상')
