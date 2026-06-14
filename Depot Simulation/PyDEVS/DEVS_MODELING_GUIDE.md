# DEVS PDF Modeling Guide

이 문서는 새 DEVS 프로젝트에서 다이어그램 PDF만 보고 모델링할 때 따르는 작업 기준이다.
특정 PDF나 기존 예시 프로젝트에 묶이지 않고, PDF마다 모델 이름과 포트 이름을 추출해 같은 방식으로 구조를 만든다.

## 기본 원칙

- PDF 다이어그램을 먼저 확인하고, 모델 이름과 포트 이름, 연결 방향을 추출한다.
- `modeling/model` 폴더는 건드리지 않는다.
- 기존 `modeling/simulation` 구조로 되돌리지 않는다. 시스템 모델은 `Control_system`, `Physical_system`으로 분리한다.
- 새 모델링 작업은 `modeling/experiment`, `modeling/Control_system`, `modeling/Physical_system` 안에서 진행한다.
- 하나의 `atomic.py`에 모든 원자모델을 넣지 않는다.
- 원자모델은 각각 별도 파일로 만든다.
- 원자모델 내부 동작 코드는 사용자가 요청하지 않으면 작성하지 않는다.
- 원자모델에는 상태 변수, 입력 포트, 출력 포트, 기본 DEVS 함수 베이스라인만 둔다.
- 기본 원자모델은 외부입력을 `WAIT` 상태에서만 처리한다. 출력 상태에서 들어온 입력을 임시 queue에 넣지 않는다.
- queue는 MCS의 scheduling 대기열처럼 도메인 의미가 있을 때만 둔다. transient 출력 상태를 처리하기 위한 `pending_jobs`는 기본 구조에 넣지 않는다.
- 빈 `__init__.py`는 프로젝트 기존 스타일에 없으면 만들지 않는다.
- `outmost.py`의 커플링은 한 줄 호출 형태로 작성해서 전체 연결 흐름이 바로 보이게 한다.

## 권장 폴더 구조

```text
modeling/
  experiment/
    experimental_frame.py
    atomic/
      <experiment_atomic_model>.py
      <generator_or_source>.py
      <analyzer_or_sink>.py
  Control_system/
    control_system.py
    atomic/
      <control_atomic_model>.py
  Physical_system/
    physical_system.py
    atomic/
      <physical_atomic_model>.py
    data/
      <physical_map_or_resource>.json
  message/
    message.py
  map/
    <map_generator_or_visualizer>.py
```

프로젝트마다 원자모델 파일명은 PDF의 모델명에 맞춰 바꾼다.

## 새 PDF 적용 절차

새 PDF를 받으면 현재 코드의 모델명을 기준으로 맞추지 말고 PDF 자체를 기준으로 모델 목록과 커플링 목록을 먼저 만든다.

1. PDF에서 상위 영역을 찾는다.
   - 실험/환경/입력/분석 영역은 `ExperimentalFrame` 후보로 본다.
   - 제어 영역은 `ControlSystem` 후보로 본다.
   - 물리/설비/업무 처리 영역은 `PhysicalSystem` 후보로 본다.
2. PDF의 박스 하나를 atomic model 후보로 기록한다.
   - 박스 이름은 class 이름의 기준이 된다.
   - 박스 안에 적힌 `to_`, `from_` 라벨은 input/output port 후보가 된다.
3. 화살표 하나를 coupling 후보로 기록한다.
   - 화살표 시작점은 source model과 source output port다.
   - 화살표 끝점은 target model과 target input port다.
4. 모델을 experiment, control, physical atomic으로 분류한다.
   - Generator, Source, Input, Analyzer, Transducer, Sink는 보통 experiment atomic이다.
   - MCS, ACS, Scheduler, Dispatcher, Controller는 보통 control atomic이다.
   - AMR, Processor, Manufacturing Manager, Worker, Resource, Device는 보통 physical atomic이다.
   - PDF에 영역 박스가 있으면 영역 구분을 우선한다.
5. 원자모델 파일을 각각 생성한다.
   - 하나의 `atomic.py`에 몰아넣지 않는다.
   - 각 파일에는 하나의 atomic class와 포트 정의만 둔다.
6. 메시지는 `modeling/message/message.py`의 `Message` 객체를 기준으로 전달한다.
7. `experimental_frame.py`에 experiment atomic들을 생성하고 외부 포트를 연결한다.
8. `control_system.py`와 `physical_system.py`에 각 영역 atomic들을 생성하고 내부/외부 포트를 연결한다.
9. `outmost.py`에서 `ExperimentalFrame`, `ControlSystem`, `PhysicalSystem` 사이의 boundary port만 연결한다.
10. PDF의 포트명과 코드의 포트명을 대조해 source/output, target/input 방향이 맞는지 확인한다.

## 이름 변환 규칙

PDF의 표시 이름을 코드 이름으로 바꿀 때는 의미를 보존하고, 현재 프로젝트 이름에 억지로 맞추지 않는다.

```text
PDF model name: Schedule manager (MCS)
file name: mcs.py 또는 schedule_manager.py
class name: MCS 또는 ScheduleManager
variable name: mcs 또는 schedule_manager
```

- 파일명과 변수명은 `snake_case`를 쓴다.
- 클래스명은 `PascalCase`를 쓴다.
- PDF에 `MCS`, `ACS`, `AMR`처럼 약어가 명확하면 클래스명은 약어를 그대로 써도 된다.
- 포트명은 PDF 라벨을 우선 사용한다.
- 같은 연결에서 coupled boundary port와 atomic 내부 port 이름이 다를 수 있다. 이 경우 `addExternalInputCoupling`, `addExternalOutputCoupling`에서 명시적으로 매핑한다.

## 커플링 생성 규칙

PDF의 화살표를 코드 coupling으로 옮길 때는 화살표가 어느 경계를 통과하는지 먼저 판단한다.

같은 coupled model 안의 atomic끼리 연결되는 경우:

```python
self.addInternalCoupling(source_model, 'output_port', target_model, 'input_port')
```

coupled model 바깥에서 들어온 값을 내부 atomic으로 넘기는 경우:

```python
self.addInputPort('boundary_input_port')
self.addExternalInputCoupling('boundary_input_port', target_model, 'target_input_port')
```

내부 atomic의 출력을 coupled model 바깥으로 내보내는 경우:

```python
self.addOutputPort('boundary_output_port')
self.addExternalOutputCoupling(source_model, 'source_output_port', 'boundary_output_port')
```

coupled model 사이를 연결하는 경우:

```python
self.addInternalCoupling(source_coupled, 'output_port', target_coupled, 'input_port')
```

커플링을 작성할 때는 다음 순서를 유지한다.

1. source model 인스턴스
2. source output port
3. target model 인스턴스
4. target input port

## 작업 절차

1. PDF에서 전체 Coupled 구조를 읽는다.
2. 모델을 `ExperimentalFrame`, `ControlSystem`, `PhysicalSystem`으로 나눈다.
3. Generator, Analyzer, Transducer처럼 실험/평가용 모델은 `modeling/experiment/atomic`에 둔다.
4. 제어 모델은 `modeling/Control_system/atomic`에 둔다.
5. 물리/설비/처리 모델은 `modeling/Physical_system/atomic`에 둔다.
6. `experimental_frame.py`에서 experiment 원자모델들을 외부 포트로 연결한다.
7. `control_system.py`와 `physical_system.py`에서 각 영역 원자모델들을 내부 커플링으로 연결한다.
8. `outmost.py`에서는 `ExperimentalFrame`, `ControlSystem`, `PhysicalSystem`만 생성하고 영역 사이 boundary port를 연결한다.
9. PDF의 `to_`, `from_` 포트 방향을 유지해서 source/output과 target/input을 맞춘다.
10. 물리 지도나 위치 데이터는 `modeling/Physical_system/data`에 두고, 지도 생성/시각화 도구는 `modeling/map`에 둔다.
11. 변경 후 `modeling/model`에 수정이 생겼는지 확인하고, 생겼으면 되돌린다.

## Coupled 파일 코드상 역할

`experimental_frame.py`는 실험용 원자모델을 감싸는 coupled model이다.
Generator, Analyzer, Transducer처럼 시스템 외부에서 입력을 만들거나 결과를 받는 원자모델을 생성하고 `addModel`로 등록한다.
ControlSystem 또는 PhysicalSystem으로 나가는 값은 `addOutputPort`와 `addExternalOutputCoupling`으로 외부 출력 포트에 연결한다.
ControlSystem 또는 PhysicalSystem에서 들어오는 값은 `addInputPort`와 `addExternalInputCoupling`으로 Analyzer 같은 내부 원자모델에 전달한다.

```text
Generator atomic model -> ExperimentalFrame output port -> Outmost -> ControlSystem/PhysicalSystem
ControlSystem/PhysicalSystem -> Outmost -> ExperimentalFrame input port -> Analyzer atomic model
```

`control_system.py`는 제어 시스템 구성요소를 감싸는 coupled model이다.
MCS, ACS 같은 제어 원자모델을 생성하고 `addModel`로 등록한 뒤, 외부에서 들어오는 입력 포트와 외부로 내보낼 출력 포트를 정의한다.

`physical_system.py`는 물리 시스템 구성요소를 감싸는 coupled model이다.
AMR, Manufacturing Manager, Worker 같은 물리/처리 원자모델을 생성하고 `addModel`로 등록한 뒤, 외부에서 들어오는 입력 포트와 외부로 내보낼 출력 포트를 정의한다.
영역 내부 구성요소끼리 주고받는 메시지는 `addInternalCoupling`으로 연결한다.

```text
ControlSystem input port -> control atomic model
control atomic model -> ControlSystem output port
PhysicalSystem input port -> physical atomic model
physical atomic model -> PhysicalSystem output port
```

`outmost.py`는 최상위 coupled model이다.
`ExperimentalFrame`, `ControlSystem`, `PhysicalSystem`만 생성하고 `addModel`로 등록한다.
원자모델을 직접 생성하거나 내부 시스템 연결을 작성하지 않는다.
세 coupled model 사이의 포트만 `addInternalCoupling`으로 연결해서 전체 입출력 흐름을 완성한다.
커플링은 한 줄 호출 형태로 작성해서 전체 연결 흐름이 바로 보이게 한다.

```text
ExperimentalFrame -> ControlSystem
ExperimentalFrame -> PhysicalSystem
ControlSystem <-> PhysicalSystem
ControlSystem/PhysicalSystem -> ExperimentalFrame
```

## 원자모델 베이스라인

원자모델은 각 파일에 하나의 클래스만 둔다.
기본 구조는 `WAIT` 상태에서만 외부입력을 받고, 출력 상태는 `ta = 0`으로 즉시 출력한 뒤 내부전이에서 `WAIT`로 돌아가는 형태다.
MCS처럼 domain queue가 필요한 경우를 제외하면 `INFO`, `WORK`, `RESULT`, `SIGNAL` 같은 출력/작업 상태에서 외부입력을 별도로 queue에 넣지 않는다.

```python
from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message

INF = float('inf')


class ExampleAtomicModel(DEVSAtomicModel):

    def __init__(self, ID):
        super().__init__(ID)

        self.addStateVariable('state', 'WAIT')

        self.addInputPort('input_from_source')
        self.addOutputPort('output_to_target')

        self.active_job = None

    def funcExternalTransition(self, strPort, event):
        state = self.getStateValue('state')
        if state == 'WAIT':
            if strPort == 'input_from_source':
                message = event if isinstance(event, Message) else Message.from_dict(event)
                self.active_job = {
                    'state': 'OUTPUT',
                    'message': message,
                }
                self.setStateValue('state', 'OUTPUT')

    def funcOutput(self):
        state = self.getStateValue('state')
        if state == 'OUTPUT':
            self.addOutputEvent('output_to_target', self.active_job['message'])

    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            pass
        elif state == 'OUTPUT':
            self.active_job = None
            self.setStateValue('state', 'WAIT')

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'WAIT':
            return INF
        elif state == 'OUTPUT':
            return 0
        return INF

    def funcSelect(self):
        return self
```

원자모델 상태 작성 기준:

- `funcExternalTransition()`은 먼저 `state = self.getStateValue('state')`를 읽고 `if state == 'WAIT':` 안에서 포트별 입력을 처리한다.
- `funcOutput()`은 현재 상태에 따라 출력 포트 하나 또는 여러 개에 이벤트를 내보낸다.
- `funcInternalTransition()`은 MCS 스타일로 `state`를 먼저 읽고 `WAIT`, 출력 상태, 작업 상태를 명시적으로 분기한다.
- `funcTimeAdvance()`는 `WAIT`에서 `INF`, 즉시 출력 상태에서 `0`, 실제 작업 시간이 있는 상태에서 해당 작업 시간을 반환한다.
- `continueTimeAdvance()`는 기본 구조에 넣지 않는다.
- 입력 메시지를 받은 뒤 출력까지 보관해야 하면 `active_job` 또는 `_pending_*` 변수에 저장한다.

## Coupled 작성 기준

`ExperimentalFrame`은 실험 환경의 외부 포트를 만든다.

```python
self.addOutputPort('event_to_system')
self.addInputPort('result_from_system')

self.addExternalOutputCoupling(generator, 'event_to_system', 'event_to_system')
self.addExternalInputCoupling('result_from_system', analyzer, 'result_from_system')
```

`ControlSystem`과 `PhysicalSystem`은 실제 시스템 내부 연결을 담당한다.

```python
self.addInputPort('event_from_generator')
self.addOutputPort('result_to_analyzer')

self.addExternalInputCoupling('event_from_generator', model_a, 'event_from_generator')
self.addExternalOutputCoupling(model_b, 'result_to_analyzer', 'result_to_analyzer')

self.addInternalCoupling(model_a, 'output_to_b', model_b, 'input_from_a')
```

`Outmost`는 상위 coupled model만 연결한다.
커플링은 아래처럼 한 줄 호출로 작성한다.

```python
EF = ExperimentalFrame('Experimental Frame')
CS = ControlSystem('Control system')
PS = PhysicalSystem('Physical system')

self.addModel(EF)
self.addModel(CS)
self.addModel(PS)

self.addInternalCoupling(EF, 'event_to_control', CS, 'event_from_experiment')
self.addInternalCoupling(CS, 'event_to_physical', PS, 'event_from_control')
self.addInternalCoupling(PS, 'result_to_analyzer', EF, 'result_from_physical')
```

## Depot Diagram_RH 적용 예

`Depot Diagram_RH.pdf` 기준 분리는 다음과 같다.

Experiment:
- `SupplyOrderGenerator`
- `ItemGenerator`
- `Analyzer`

Control system:
- `MCS`
- `ACS`

Physical system:
- `AMR`
- `ManufacturingManager`
- `TransportationManager`
- `Worker`

주요 연결:
- `SupplyOrderGenerator.supply_order_to_MCS -> ControlSystem.supply_order_to_MCS -> MCS.supply_order_from_gen`
- `ItemGenerator.item_update_to_mm -> PhysicalSystem.item_update_to_mm -> ManufacturingManager.item_update_from_gen`
- `MCS.m_req_to_mm -> ManufacturingManager.m_req_from_MCS`
- `ManufacturingManager.m_info_to_MCS -> MCS.m_info_from_mm`
- `Worker.item_depot_update_to_MCS -> MCS.item_depot_update_from_wk`
- `MCS.schedule_info_to_ACS -> ACS.schedule_info_from_MCS`
- `ACS.amr_req_to_tm -> TransportationManager.amr_req_from_ACS`
- `ACS.amr_update_to_tm -> TransportationManager.amr_update_from_ACS`
- `TransportationManager.amr_info_to_ACS -> ACS.amr_info_from_tm`
- `ACS.cmd_to_amr -> AMR.cmd_from_ACS`
- `AMR.cmd_completed_to_mm -> ManufacturingManager.cmd_completed_from_amr`
- `AMR.return_completed_to_tm -> TransportationManager.return_completed_from_amr`
- `ManufacturingManager.arrive_to_wk -> Worker.arrive_from_mm`
- `ACS.operation_result_to_analyzer -> Analyzer.operation_result_from_ACS`
- `AMR.physical_result_to_analyzer -> Analyzer.physical_result_from_amr`
- `ManufacturingManager.work_info_to_analyzer -> Analyzer.work_info_from_mm`

현재 Physical system 원자모델 상태 흐름:

```text
ManufacturingManager
WAIT
  ? m_req_from_MCS        -> INFO   -> ! m_info_to_MCS          -> WAIT
  ? item_update_from_gen  -> INFO   -> ! m_info_to_MCS          -> WAIT
  ? cmd_completed_from_amr -> RESULT -> ! work_info_to_analyzer -> WAIT
  ? cmd_completed_from_amr -> RESULT -> ! work_info_to_analyzer -> SIGNAL -> ! arrive_to_wk -> WAIT

TransportationManager
WAIT
  ? amr_req_from_ACS          -> INFO -> ! amr_info_to_ACS -> WAIT
  ? amr_update_from_ACS       -> AMR 가용 목록 갱신 후 WAIT 유지
  ? return_completed_from_amr -> AMR 가용 목록 갱신 후 WAIT 유지

Worker
WAIT
  ? arrive_from_mm -> WORK
WORK
  ta = 대차 내 part 수량 * seconds_per_part
  ! item_depot_update_to_MCS
  -> WAIT
```

`outmost.py` 실제 커플링:

```python
self.addInternalCoupling(EF, 'supply_order_to_MCS', CS, 'supply_order_to_MCS')
self.addInternalCoupling(EF, 'item_update_to_mm', PS, 'item_update_to_mm')
self.addInternalCoupling(CS, 'm_req_to_mm', PS, 'm_req_to_mm')
self.addInternalCoupling(PS, 'm_info_to_MCS', CS, 'm_info_to_MCS')
self.addInternalCoupling(PS, 'item_depot_update_to_MCS', CS, 'item_depot_update_to_MCS')
self.addInternalCoupling(CS, 'amr_req_to_tm', PS, 'amr_req_to_tm')
self.addInternalCoupling(CS, 'amr_update_to_tm', PS, 'amr_update_to_tm')
self.addInternalCoupling(PS, 'amr_info_to_ACS', CS, 'amr_info_to_ACS')
self.addInternalCoupling(CS, 'cmd_to_amr', PS, 'cmd_to_amr')
self.addInternalCoupling(CS, 'operation_result_to_analyzer', EF, 'operation_result_to_analyzer')
self.addInternalCoupling(PS, 'physical_result_to_analyzer', EF, 'physical_result_to_analyzer')
self.addInternalCoupling(PS, 'work_info_to_analyzer', EF, 'work_info_to_analyzer')
```

## 통합 메시지 기준

Depot 메시지는 `modeling/message/message.py`의 `Message` 클래스를 기준으로 사용한다.
`Message`는 dataclass이며, 전체 이벤트 필드를 속성으로 갖는다.
dict 입력이 들어오면 `Message.from_dict()`로 변환하고, 기존 메시지를 갱신할 때는 `message.update()`를 사용한다.

`message.update()`는 원본을 직접 바꾸지 않고 deep copy된 새 `Message`를 반환한다.
따라서 같은 메시지 객체를 여러 출력으로 나눠 보내도 원본 이벤트가 오염되지 않는다.

```python
from modeling.message.message import Message

msg = Message(product='A', part='a', cart_count=1)
msg = msg.update(current_location='material_depot', amr_status='WORK')
self.addOutputEvent('physical_result_to_analyzer', msg)
```

dict를 받아 메시지로 바꿀 때:

```python
msg = Message.from_dict({
    'product': 'A',
    'part': 'a',
    'from': {'location_id': 'A_WH_F_a_O_1'},
    'to': {'location_id': 'A_ML_a_P1'},
})
```

`from`은 Python 예약어이므로 Python 속성에서는 `from_`를 사용한다.
단, dict/JSON으로 직렬화할 때는 다시 `from` 키로 나간다.

```python
print(msg.from_)
print(msg.to_dict()['from'])
```

현재 주요 메시지 필드:

- `product`, `part`, `cart_count`, `generated_time`
- `info_type`, `from_`, `to`
- `AMR_id`, `task_id`, `task_type`, `task_step`
- `current_location`, `goal_location`, `route`
- `amr_status`, `idle_type`, `arrival_event`, `current_situation`
- `location_id`, `location_code`, `node`, `x`, `y`, `access_node_id`
- `type`, `capacity`, `status`, `load_type`

`Message`에 정의되지 않은 키는 기본적으로 무시된다.
새 필드가 필요하면 먼저 `message.py`의 `Message` dataclass에 필드를 추가한 뒤 원자모델에서 사용한다.

## 지도 데이터 기준

Physical system에서 사용하는 실제 지도 데이터는 `modeling/Physical_system/data/map.json`이다.
`ManufacturingManager`는 WH/ML 위치를 읽어 자재데포, 공정데포, 회수데포 후보를 만든다.
`Worker`는 같은 map에서 ML의 `R` 노드를 회수데포로 읽는다.

지도 생성 또는 시각화 스크립트는 `modeling/map`에 둔다.
원자모델이 직접 사용하는 런타임 데이터는 `modeling/Physical_system/data`에 둔다.

## 완료 전 확인

- `git status --short`에서 `modeling/model` 변경이 없어야 한다.
- 새 시스템 모델이 `modeling/simulation` 아래에 만들어지지 않았는지 확인한다.
- 원자모델이 각각 별도 파일인지 확인한다.
- PDF에서 추출한 모델 목록과 생성된 atomic 파일 목록이 일치해야 한다.
- PDF에서 추출한 화살표 목록과 `add*Coupling` 목록이 일치해야 한다.
- `experimental_frame.py`, `control_system.py`, `physical_system.py`, `outmost.py`의 포트명이 서로 정확히 맞아야 한다.
- MM, TM, Worker 같은 기본 원자모델에 transient 상태용 `pending_jobs`나 busy-state queue가 들어가지 않았는지 확인한다.
- 새 메시지 필드를 사용했다면 `modeling/message/message.py`의 `Message` dataclass에 정의되어 있는지 확인한다.
- Python 실행 환경이 있으면 import 또는 컴파일 검사를 실행한다.
