---
args: {"update": "DispatchUpdate", "prior": "DeliveryRecoveryBook"}
returns: "RoutingDelta"
---
Read the notices as scoped changes to route, carrier service, interval and shipment class. Compare the supplied prior impact to actual new evidence rather than adopting it as current truth. A closure, delay, reduced capacity and partial reopening are different states. Record uncertain reopening times as uncertainty, not supply. Return only the leg/reservation dependencies and item classes supported by these source IDs.
