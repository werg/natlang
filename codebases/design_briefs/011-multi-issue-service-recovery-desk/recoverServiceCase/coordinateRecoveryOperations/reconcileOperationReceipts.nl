---
description: Determine whether an external operation is confirmed, failed, unknown or safely retryable.
args:{remedies:RemedyBundle[];receipts:EvidencePage[]}
returns:OperationReconciliation[]
---
Match receipts to exact operation ID, action kind and target. A success receipt confirms only its named effect; absence or timeout is unknown. Mark retry safe only when authoritative status proves no effect occurred or an earlier failure receipt permits retry. Preserve expired reservation as a distinct result.
