---
args: { deliverableId: ID, eligibleReceipts: HandoverAcceptanceReceipt[] }
returns: DeliverableAcceptanceTrace
---
Inspect the same-edition, authority-validated receipts supplied by the parent. For this deliverable only, retain receipt IDs that explicitly list it in acceptedDeliverableIds. Preserve each exception concerning its delivery or usability. If exception wording is too broad to exclude this item, keep that uncertainty rather than assigning the exception to a convenient other document. A later signature without an explicit resolution cannot erase an earlier unresolved exception.

Return DeliverableAcceptanceTrace { deliverableId, covered: at least one exact item acceptance exists and no unresolved exception affects it, receiptIds: those actual item receipt IDs, blockers: the missing acceptance or unresolved exception evidence }. A receipt for another deliverable provides no coverage here.
