---
description: Exact counterfactual gate for recipient-specific authorization and date/category scope.
args: { evidence: EvidenceSlice; draft: DraftRecordAuthorization }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when authorized recipient, permitted categories, patient and date window are supported by current authorization. Fail when the requested packet exceeds the recorded scope or authorization has expired. Return needs-review when authority, expiry or recipient binding is missing. Return only the finite verdict.
