from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from modeling.experiment.atomic.analyzer import Analyzer
from modeling.experiment.atomic.item_generator import ItemGenerator
from modeling.experiment.atomic.supply_order_generator import SupplyOrderGenerator


class ExperimentalFrame(DEVSCoupledModel):

    def __init__(self, ID):
        super().__init__(ID)

        # Models
        supply_order_generator = SupplyOrderGenerator('Supply Order Generator')
        item_generator = ItemGenerator('Item Generator')
        analyzer = Analyzer('Analyzer')

        self.addModel(supply_order_generator)
        self.addModel(item_generator)
        self.addModel(analyzer)

        # Output ports
        self.addOutputPort('supply_order_to_MCS')
        self.addOutputPort('item_update_to_mm')

        # Input ports
        self.addInputPort('operation_result_to_analyzer')
        self.addInputPort('physical_result_to_analyzer')
        self.addInputPort('work_info_to_analyzer')

        # External output coupling
        self.addExternalOutputCoupling(supply_order_generator, 'supply_order_to_MCS', 'supply_order_to_MCS')
        self.addExternalOutputCoupling(item_generator, 'item_update_to_mm', 'item_update_to_mm')

        # External input coupling
        self.addExternalInputCoupling('operation_result_to_analyzer', analyzer, 'operation_result_from_ACS')
        self.addExternalInputCoupling('physical_result_to_analyzer', analyzer, 'physical_result_from_amr')
        self.addExternalInputCoupling('work_info_to_analyzer', analyzer, 'work_info_from_mm')
