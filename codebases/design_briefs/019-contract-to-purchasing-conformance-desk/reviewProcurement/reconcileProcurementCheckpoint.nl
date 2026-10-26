---
args: { prior: ProcurementCheckpoint|null, request: ProcurementReviewRequest, agreementFile: AgreementDeskFile, intent: StageResultPurchaseIntent, agreement: StageResultAgreementScope|null, options: StageResultPurchasingOption|null, disposition: StageResultPurchaseDisposition|null }
returns: ProcurementCheckpoint
---
Bind each purchase need to the catalog edition and exact legal entity. Preserve agreement lines, accepted commitments and cumulative approval receipts as separate records. If a catalog version or agreement term expires, reopen only pending options that rely on that line; retain completed historical commitments. Bundle removal changes the price basis and requires its own exact repricing evidence. Duplicate commitment or approval receipts are no-ops. Never treat a selected option as a released order.
