from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
from config import ITEM_UPDATES


INF = float('inf')


class ItemGenerator(DEVSAtomicModel):

    def __init__(self, ID):
        super().__init__(ID)

        self.addStateVariable('state', 'GEN')
        self.item_index = 0

        self.addOutputPort('item_update_to_mm')

    def funcExternalTransition(self, strPort, event):
        pass


    def funcOutput(self):
        state = self.getStateValue('state')
        if state == 'GEN' and self.item_index < len(ITEM_UPDATES):
            order = ITEM_UPDATES[self.item_index]
            msg = Message(
                product=order['product'],
                part=order['part'],
                cart_count=order['cart_count'],
                generated_time=f"{self.getTime():08.2f}",
            )
            print(f"[{self.getTime():08.2f}] ItemGenerator outputs item_update_to_mm: {msg.product} {msg.part} x{msg.cart_count}")
            self.addOutputEvent('item_update_to_mm', msg)


    def funcInternalTransition(self):
        state = self.getStateValue('state')
        if state == 'GEN':
            self.item_index += 1
            if self.item_index < len(ITEM_UPDATES):
                self.setStateValue('state', 'GEN')
            else:
                self.setStateValue('state', 'SIM-DONE')
        

    def funcTimeAdvance(self):
        state = self.getStateValue('state')
        if state == 'GEN':
            return INF
        if state == 'SIM-DONE':
            return INF
        

    def funcSelect(self):
        pass
