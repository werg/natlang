---
description: Verify a candidate response against an exact tender-bound deterministic bundle validation artifact.
args:{request:BidPreparationRequest;constraints:EvidencePage[];sections:ResponseSection[];validations:PortalBundleValidation[]}
returns:TenderGuardrail
---
Find one validator artifact matching tender ID, exact response hash, current addendum revision and response sections. Require status pass, complete required forms and exact filename, count, size and format limits. If artifact is missing or mismatched, return blocked/unknown; do not calculate encoded size or claim a portal check occurred. Preserve file-set hash for approval.
