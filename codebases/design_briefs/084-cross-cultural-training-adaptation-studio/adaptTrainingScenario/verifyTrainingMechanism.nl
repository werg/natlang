---
description: Exact counterfactual gate for preserving training objective while adapting incidental context.
args: { evidence: EvidenceSlice; draft: DraftTrainingMechanism }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when competency, required decision and incidental elements are grounded in the supplied curriculum. Fail when the proposed mechanism changes the learning objective or turns an informal analogy into a required competency. Return needs-review when audience assumptions or source curriculum are disputed. Return only the finite verdict.
