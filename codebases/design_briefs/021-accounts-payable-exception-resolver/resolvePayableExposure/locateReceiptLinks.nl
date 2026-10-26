---
args: {"line": "PayableLine", "documents": "EvidencePage[]", "reservations": "ReceiptReservation[]"}
returns: "ReceiptLink[]"
---
Enumerate candidate order and receipt lines by supplier reference, delivery context, descriptions and compatible units. For each candidate call distinguishReceiptSupport(line, candidateEvidence, reservations). Permit several receipts per invoice line and one receipt across several lines only within unconsumed quantity. Return alternatives and the evidence needed to distinguish them. Never infer physical acceptance from the existence of a receipt header.
