---
description: Exact counterfactual gate for item-specific rights, channel, attribution and expiry.
args: { evidence: EvidenceSlice; draft: DraftCollectionPermission }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when permitted use, prohibited use, attribution rule and expiry are supported by the rights holder's current authority. Fail when object ownership is treated as reproduction permission or print rights as online clearance. Return needs-review when right type, holder, channel or expiry is missing. Return only the finite verdict.
