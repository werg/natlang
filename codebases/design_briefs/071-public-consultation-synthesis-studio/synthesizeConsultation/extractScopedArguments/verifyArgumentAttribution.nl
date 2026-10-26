---
description: Exact counterfactual gate for argument attribution and preservation of mixed submission positions.
args: { evidence: EvidenceSlice; draft: DraftConsultationArgument }
returns: "pass" | "fail" | "needs-review"
readout: decision
---
Pass when each position, reason, factual claim and proposed action in `draft.values` is traceable to the identified submission context and conditional or mixed views are preserved. Fail when the draft invents a motive, attributes another participant's reason, or erases one part of a mixed submission. Return needs-review when source context, submission identity or current evidence is missing. Return only the finite verdict.
