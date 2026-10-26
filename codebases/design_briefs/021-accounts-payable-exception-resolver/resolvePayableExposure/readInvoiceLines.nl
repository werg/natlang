---
args: {"documents": "EvidencePage[]", "versions": "LiabilityVersion[]"}
returns: "PayableLine[]"
---
Partition each active invoice into original line identities, descriptions, quantities, taxes, freight and credits. Preserve exact integer minor units and stated currency/scale, converting textual decimal amounts mechanically only when their digits and scale are explicit. Keep unreadable values null with a blocker. Compare line totals to the invoice total using sumMinorUnits for one currency/scale at a time; a mismatch remains unresolved and cannot be filled with an invented balancing line.
