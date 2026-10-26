---
args: { replay: PairedCompatibilityReplay[], consumers: BehaviorConsumerMap[], release: CompatibilityReleaseRequest }
returns: MigrationWavePlan
---
Group consumers by shared operational prerequisite and behavior risk, not package name alone. Preserve an old-version support window when a customer is pinned. Do not schedule an unowned integration before ownership is assigned. For each wave state entry evidence, responsible owner, and a tested rollback trigger.
For each proposed wave bind rehearsal = verifyWaveRollbackEvidence(wave, release.rollbackDataPlan). Keep the wave blocked if rehearsal evidence is absent.
