---
args: {"alternatives": "LegAlternative[]", "orders": "ShippingOrder[]", "reservations": "CarrierReservation[]", "priorityRules": "string[]"}
returns: "LegAlternative[]"
---
Carry a residual capacity ledger for each exact offer version. Consider supplied-priority orderings and whole-set alternatives in NatLang, keeping unsatisfied demand visible. Subtract confirmed, requested and unknown occupied quantities exactly; release only confirmed cancellations. Produce provisional reservation operation IDs with expected offer versions. Two concurrent plans claiming the same final slot cannot both become supported: actual atomic reservation receipts decide which is available, and the loser must replan from the changed capacity.
