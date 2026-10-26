---
args: {"partitions": "BalancePartition[]", "versions": "LiabilityVersion[]", "reservations": "ReceiptReservation[]", "dossier": "PayableDossier"}
returns: "PaymentInstruction[]"
---
Group supported payable amounts by legal entity, liability version, currency and approved posting period. Call chooseIndependentApproval(entity, dossier.submitter, amount, dossier.delegations) for each group. Require exact receipt-reservation receipts before marking an instruction approved. Bind the instruction to version, amount, reservation identities and source dates. Reuse an existing operation identity for unchanged retries; a new approved amount requires an invalidated predecessor and a fresh instruction. Return review instructions with explicit missing authority where necessary.
