---
args: {"observations": "DeviceObservation[]", "prior": "StockIdentityDecision[]", "evidence": "Evidence[]"}
returns: "StockIdentityDecision[]"
---
For each admitted observation call compareLotSerialEvidence(observation, prior, evidence). Bind identity from compatible serial, lot, packaging, movement and location facts. Two visually identical cartons with distinct lots remain distinct. A mislabeled item and a misplaced correctly labeled item have different remedies. Retain several candidates and request the scan that would distinguish them; do not select the nearest description.
