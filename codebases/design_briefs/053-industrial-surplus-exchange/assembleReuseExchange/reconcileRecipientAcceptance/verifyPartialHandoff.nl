---
args: {"receipt": "TransferObservation", "legs": "ExchangeLeg[]", "profiles": "MaterialProfile[]"}
returns: "string[]"
---
Verify leg, material/lot, quantity, condition, title transfer and recipient authority independently. Custody by a courier is not recipient acceptance. Require accepted plus rejected plus still-pending quantities to reconcile exactly to the actual handoff. A partially accepted lot leaves residual material available only after confirmed release of its reservation; otherwise preserve pending custody. Return missing or contradictory receipt evidence rather than inventing acceptance.
