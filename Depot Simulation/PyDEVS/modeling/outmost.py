from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from SimulationEngine.Utility.Configurator import Configurator
from modeling.Control_system.control_system import ControlSystem
from modeling.Physical_system.physical_system import PhysicalSystem
from modeling.experiment.experimental_frame import ExperimentalFrame


class Outmost(DEVSCoupledModel):

    def __init__(self, ID='Outmost', objConfiguration=None):
        if objConfiguration is None and isinstance(ID, Configurator):
            objConfiguration = ID
            ID = 'Outmost'

        super().__init__(ID)

        self.objConfiguration = objConfiguration

        # Models
        EF = ExperimentalFrame('Experimental Frame')
        CS = ControlSystem('Control system', self.objConfiguration)
        PS = PhysicalSystem('Physical system', self.objConfiguration)

        self.addModel(EF)
        self.addModel(CS)
        self.addModel(PS)

        # Input Ports

        # Output Ports

        # External Input Coupling

        # External Output Coupling

        # Internal Coupling
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

        # Variables
