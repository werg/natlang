---
args: {"observation": "DeviceObservation", "prior": "StockIdentityDecision[]", "evidence": "Evidence[]"}
returns: "StockIdentityDecision"
---
Compare observed serial and lot bytes with verified custody history and prior labels. Preserve every label before relabeling and prevent an item-level match from erasing lot identity. A contradictory serial prevents a confirmed match; an unreadable serial yields a specific inspection task. Return source-backed alternatives and exact missing evidence without altering the physical stock ledger.
