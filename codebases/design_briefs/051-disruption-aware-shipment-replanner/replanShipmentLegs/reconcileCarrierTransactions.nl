---
args: {"prior": "CarrierReservation[]", "notices": "Evidence[]", "receipts": "CarrierReservation[]"}
returns: "CarrierReservation[]"
---
For each known carrier operation call interpretCarrierOutcome(operation, notices, receipts). Match carrier, operation identity, offer version, shipment class and quantity before applying a receipt. A later item-class rejection supersedes an earlier acceptance for that booking's feasibility. Keep cancellation requests occupied until the carrier confirms cancellation; retrying an unknown booking is forbidden until authoritative readback determines its outcome.
