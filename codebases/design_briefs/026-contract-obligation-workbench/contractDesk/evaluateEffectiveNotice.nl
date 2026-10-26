---
args: { request: ObligationReviewRequest, receipts: EvidencePage[] }
returns: ObligationGuardrail
---
Track dispatch event, destination address, delivery or rejection receipt, and contractual notice condition independently. A message sent to the wrong address may be dispatched but is not effective contractual notice unless the agreed rule says otherwise.
