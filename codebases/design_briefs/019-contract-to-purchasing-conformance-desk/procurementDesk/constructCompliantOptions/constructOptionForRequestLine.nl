---
args: { intent: PurchaseIntent, agreement: AgreementScope, finding: ConformanceFinding, bundleReprice: ProcurementGuardrail }
returns: PurchasingOption[]
---
Construct only alternatives satisfying this request line's confirmed essential constraints and the exact agreement legal entity, territory and effective period. A conditional clause remains conditional until its evidence is supplied. If removal from a bundle changed the line price or minimum quantity, consume the new repricing evidence; never copy the old bundle price. If no compliant item remains, return no accepted alternative and preserve the blocker for an authorized exception route.
