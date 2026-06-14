from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from modeling.Control_system.atomic.acs import ACS
from modeling.Control_system.atomic.mcs import MCS


class ControlSystem(DEVSCoupledModel):

    def __init__(self, ID, objConfiguration=None):
        super().__init__(ID)
        self.objConfiguration = objConfiguration

        # Models
        mcs = MCS('MCS')
        acs = ACS('ACS', objConfiguration=self.objConfiguration)

        self.addModel(mcs)
        self.addModel(acs)

        # Input ports
        self.addInputPort('supply_order_to_MCS')
        self.addInputPort('m_info_to_MCS')
        self.addInputPort('item_depot_update_to_MCS')
        self.addInputPort('amr_info_to_ACS')

        # Output ports
        self.addOutputPort('m_req_to_mm')
        self.addOutputPort('cmd_to_amr')
        self.addOutputPort('operation_result_to_analyzer')
        self.addOutputPort('amr_req_to_tm')
        self.addOutputPort('amr_update_to_tm')

        # External input couplings
        self.addExternalInputCoupling('supply_order_to_MCS', mcs, 'supply_order_from_gen')
        self.addExternalInputCoupling('m_info_to_MCS', mcs, 'm_info_from_mm')
        self.addExternalInputCoupling('item_depot_update_to_MCS', mcs, 'item_depot_update_from_wk')
        self.addExternalInputCoupling('amr_info_to_ACS', acs, 'amr_info_from_tm')

        # External output couplings
        self.addExternalOutputCoupling(mcs, 'm_req_to_mm', 'm_req_to_mm')
        self.addExternalOutputCoupling(acs, 'cmd_to_amr', 'cmd_to_amr')
        self.addExternalOutputCoupling(acs, 'operation_result_to_analyzer', 'operation_result_to_analyzer')
        self.addExternalOutputCoupling(acs, 'amr_req_to_tm', 'amr_req_to_tm')
        self.addExternalOutputCoupling(acs, 'amr_update_to_tm', 'amr_update_to_tm')

        # Internal couplings
        self.addInternalCoupling(mcs, 'schedule_info_to_ACS', acs, 'schedule_info_from_MCS')
