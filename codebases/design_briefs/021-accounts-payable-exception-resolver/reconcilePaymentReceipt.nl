---
args: {"ledger": "PayableBook", "observation": "PostingObservation"}
returns: "PostingReconciliation"
---
Locate the existing instruction by observation.operationId before considering any new action. Call comparePostingEvidence(instruction, observation). If processor and ledger disagree or either is missing, keep the instruction unknown and return an investigation; do not retry payment. On an exact posted match attach both receipt IDs, mark only its reserved receipt quantities consumed, and reduce the remaining obligation exactly once. A verified not-posted result permits a separately authorized release or retry under the same operation identity. Return an incremented ledger with all original service dates and credit history intact.
