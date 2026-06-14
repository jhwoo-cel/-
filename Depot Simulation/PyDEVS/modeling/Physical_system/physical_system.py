from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from modeling.Physical_system.atomic.amr import AMR
from modeling.Physical_system.atomic.manufacturing_manager import ManufacturingManager
from modeling.Physical_system.atomic.transportation_manager import TransportationManager
from modeling.Physical_system.atomic.worker import Worker
from config import AMR_SPEED_M_PER_SEC


class PhysicalSystem(DEVSCoupledModel):

    def __init__(self, ID, objConfiguration=None):
        super().__init__(ID)
        self.objConfiguration = objConfiguration

        # Models
        manufacturing_manager = ManufacturingManager('Manufacturing manager')
        transportation_manager = TransportationManager('Transportation manager', self.objConfiguration)
        worker = Worker('Worker')


        self.addModel(manufacturing_manager)
        self.addModel(transportation_manager)
        self.addModel(worker)

        # =========================
        # AMR 여러 대 생성
        # =========================
        self.amrs = []

        amr_configs = []
        if self.objConfiguration is not None:
            amr_configs = self.objConfiguration.getConfiguration("AMR_LIST") or []

        # 혹시 AMR_LIST가 없으면 기본 AMR 1대 생성
        if len(amr_configs) == 0:
            amr_configs = [
                {
                    "AMR_id": "AMR",
                    "speed": AMR_SPEED_M_PER_SEC,
                    "current_location": "STATION_01",
                    "battery_level": 100.0,
                    "physical_log_interval": 1.0
                }
            ]

        for amr_cfg in amr_configs:
            amr = AMR(
                ID=amr_cfg["AMR_id"],
                speed=amr_cfg.get("speed", AMR_SPEED_M_PER_SEC),
                loaded_max_speed=amr_cfg.get("loaded_max_speed"),
                unloaded_max_speed=amr_cfg.get("unloaded_max_speed"),
                empty_cart_max_speed=amr_cfg.get("empty_cart_max_speed"),
                docking_speed=amr_cfg.get("docking_speed"),
                docking_distance=amr_cfg.get("docking_distance"),
            )

            amr.current_location = amr_cfg.get("current_location")
            amr.battery_level = amr_cfg.get("battery_level", 100.0)
            amr.physical_log_interval = amr_cfg.get("physical_log_interval", 1.0)
            amr.amr_product = amr_cfg.get("product")

            self.amrs.append(amr)
            self.addModel(amr)

        # Input ports
        self.addInputPort('item_update_to_mm')
        self.addInputPort('m_req_to_mm')
        self.addInputPort('cmd_to_amr')
        self.addInputPort('amr_req_to_tm')
        self.addInputPort('amr_update_to_tm')

        # Output ports
        self.addOutputPort('m_info_to_MCS')
        self.addOutputPort('item_depot_update_to_MCS')
        self.addOutputPort('physical_result_to_analyzer')
        self.addOutputPort('work_info_to_analyzer')
        self.addOutputPort('amr_info_to_ACS')

        # External input couplings
        self.addExternalInputCoupling('item_update_to_mm', manufacturing_manager, 'item_update_from_gen')
        self.addExternalInputCoupling('m_req_to_mm', manufacturing_manager, 'm_req_from_MCS')
        self.addExternalInputCoupling('amr_req_to_tm', transportation_manager, 'amr_req_from_ACS')
        self.addExternalInputCoupling('amr_update_to_tm', transportation_manager, 'amr_update_from_ACS')
        
        
        # ACS command를 모든 AMR에게 broadcast
        for amr in self.amrs:
            self.addExternalInputCoupling('cmd_to_amr', amr, 'cmd_from_ACS')

        # External output couplings
        self.addExternalOutputCoupling(manufacturing_manager, 'm_info_to_MCS', 'm_info_to_MCS')
        self.addExternalOutputCoupling(worker, 'item_depot_update_to_MCS', 'item_depot_update_to_MCS')
        self.addExternalOutputCoupling(manufacturing_manager, 'work_info_to_analyzer', 'work_info_to_analyzer')
        self.addExternalOutputCoupling(transportation_manager, 'amr_info_to_ACS', 'amr_info_to_ACS')

        # AMR 주행 로그를 PhysicalSystem output으로 전달
        for amr in self.amrs:
            self.addExternalOutputCoupling(amr, 'physical_result_to_analyzer', 'physical_result_to_analyzer')


        # Internal couplings
        for amr in self.amrs:
            self.addInternalCoupling(amr, 'cmd_completed_to_mm', manufacturing_manager, 'cmd_completed_from_amr')
            self.addInternalCoupling(amr, 'return_completed_to_tm', transportation_manager, 'return_completed_from_amr')
            self.addInternalCoupling(amr, 'physical_result_to_tm', transportation_manager, 'physical_result_from_amr')
        self.addInternalCoupling(manufacturing_manager, 'arrive_to_wk', worker, 'arrive_from_mm')
        self.addInternalCoupling(worker, 'w_req_to_mm', manufacturing_manager, 'w_req_from_wk')
        self.addInternalCoupling(manufacturing_manager, 'm_info_to_wk', worker, 'm_info_from_mm')
