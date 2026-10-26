---
description: Topologically schedule selected scene cues and verify every critical owner before stage-manager review.
args: { production: ImmersiveProduction; routes: VariantSet; resources: ResourceTimeline; showEvidence: EvidenceItem[] }
returns: CueSequence
---
1. Bind schedule = orderCueDependencies(production, routes, resources, showEvidence), then ownership = verifyCueOwners(schedule.cues, resources, showEvidence). Preserve the documented dependency order; a cue-sheet array is not itself proof of precedence.
2. Construct cues from schedule.cues with their exact identities, owners, source bindings, resources, timing windows and dependencies. Keep their state awaiting-acknowledgment. A prior acknowledgment must bind this exact cue revision before it can satisfy ownership; it still cannot establish execution.
3. Bind conflicts to the schedule's cycles and overlappingResources together with ownership.conflicts and ownership.missingOwners. Bind pendingAcknowledgments to the actual unresolved cue-owner acknowledgments. Return CueSequence { order: the scheduled cue IDs, cues, conflicts, pendingAcknowledgments, executionReceipts: [], ready }.
4. Set ready only when conflicts and pendingAcknowledgments are empty and at least one approved route is selected. The returned cue plan contains no newly asserted execution receipt. Actual execution is reconstructed independently from observed run-matched receipts.
