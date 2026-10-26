---
args: { intent: StageResultPurchaseIntent, agreement: StageResultAgreementScope, findings: StageResultConformanceFinding, bundleReprice: ProcurementGuardrail }
returns: StageResultPurchasingOption
---
Construct alternatives only from current eligible agreement items and confirmed essential constraints. When removing a line from a discounted bundle, use the recalculated price and remaining-quantity minimum; do not reuse the original bundle price. If no in-agreement option satisfies essential need, offer the exception route without weakening requirements.
