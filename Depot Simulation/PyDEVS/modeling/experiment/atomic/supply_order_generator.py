from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from modeling.message.message import Message
from config import SUPPLY_ORDER_GROUPS

INF = float('inf')


class SupplyOrderGenerator(DEVSAtomicModel):
    """Emit integrated A/B supply orders with independent product-part timers."""

    def __init__(self, ID):
        super().__init__(ID)

        self.addStateVariable('state', 'GEN')
        self.addOutputPort('supply_order_to_MCS')

        self.groups = {
            self._group_key(group): dict(group)
            for group in SUPPLY_ORDER_GROUPS
        }
        self.next_fire = {
            key: float(group["interval"])
            for key, group in self.groups.items()
        }
        self.fire_count = {key: 0 for key in self.groups}
        self.pending_messages = []
        self.firing_key = None

    def funcExternalTransition(self, strPort, event):
        pass

    def funcOutput(self):
        if self.getStateValue('state') != 'GEN':
            return

        if not self.pending_messages:
            self._prepare_next_due_messages()

        if not self.pending_messages:
            return

        msg = self.pending_messages[0]
        print(
            f"[{self.getTime():08.2f}][SupplyOrderGenerator] generated "
            f"task_id={msg.task_id}, product={msg.product}, part={msg.part}, "
            f"target_slot={msg.target_slot}, cart_count={msg.cart_count}"
        )
        self.addOutputEvent('supply_order_to_MCS', msg)

    def funcInternalTransition(self):
        if self.getStateValue('state') != 'GEN':
            return

        if self.pending_messages:
            self.pending_messages.pop(0)

        if self.pending_messages:
            return

        if self.firing_key is not None:
            group = self.groups[self.firing_key]
            self.next_fire[self.firing_key] += float(group["interval"])
            self.fire_count[self.firing_key] += 1
            self.firing_key = None

    def funcTimeAdvance(self):
        if self.getStateValue('state') != 'GEN' or not self.next_fire:
            return INF
        if self.pending_messages:
            return 0
        earliest = min(self.next_fire.values())
        return max(0.0, earliest - self.getTime())

    def funcSelect(self):
        pass

    def _prepare_next_due_messages(self):
        if not self.next_fire:
            return

        earliest = min(self.next_fire.values())
        due_keys = [
            key
            for key, fire_time in self.next_fire.items()
            if abs(fire_time - earliest) <= 1e-9
        ]
        due_keys.sort()
        self.firing_key = due_keys[0]

        group = self.groups[self.firing_key]
        sequence = self.fire_count[self.firing_key] + 1
        self.pending_messages = [
            self._message_for_target(group, target, sequence)
            for target in group["targets"]
        ]

    def _message_for_target(self, group, target, sequence):
        product = group["product"]
        part = group["part"]
        target_slot = target.get("target_slot")
        slot_suffix = f"_P{target_slot}" if target_slot is not None else ""
        task_id = f"{product}_{part}{slot_suffix}_{sequence:06d}"

        return Message(
            task_id=task_id,
            task_type="supply",
            product=product,
            part=part,
            cart_count=group["cart_count"],
            generated_time=f"{self.getTime():08.2f}",
            target_slot=target_slot,
            target_label=target.get("target_label"),
            chain_recovery=target.get("chain_recovery"),
        )

    def _group_key(self, group):
        return (group["product"], group["part"])
