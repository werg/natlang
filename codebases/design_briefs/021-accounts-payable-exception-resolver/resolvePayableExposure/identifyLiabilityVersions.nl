---
args: {"dossier": "PayableDossier", "ledger": "PayableBook"}
returns: "LiabilityVersion[]"
---
Group documents by legal entity, supplier, evidenced invoice lineage and contract reference. Call compareInvoiceRevision(candidate, prior, dossier.documents) for each possible same-number revision. A repeated number alone proves neither duplication nor independence. Preserve original service dates, currency and period. Attach verified credit notes to their specific liability/version and retain conflicting or missing lineage as blockers.
