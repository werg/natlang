---
args: { plan: ReleaseTransitionProposal, wave: CoordinatedChangeWave, receipt: ReleaseWaveReceipt }
returns: { allAccepted: boolean, unresolvedConsumerIds: ID[], reasons: string[] }
---
1. Require each wave.consumerId to have a supported, owned BehaviorConsumerMap and an actual receipt.acceptedConsumerIds entry for this candidate revision. An owner name or communication notice is not acceptance evidence.
2. For each paired compatibility replay for a wave consumer, require actual replay evidence with the same consumerId and input, and compare the observed baseline and candidate outcomes with the predeclared acceptanceCriterion. A missing or breaking observation blocks this wave.
3. Return only the unresolved consumer IDs and the mismatch reasons. Preserve pinned consumers outside this wave under the old-version support window.
