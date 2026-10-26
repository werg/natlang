---
args: { release: CompatibilityReleaseRequest }
returns: ReleaseTransitionProposal
---
1. Bind changes = compareObservedReleaseBehavior(release). Inspect defaults, ordering, retries, and error semantics even when schemas are unchanged.
2. Bind consumers = traceBehaviorConsumers(changes, release). Resolve owners and version pins per behavior; package-name overlap alone is not dependency evidence.
3. Bind replay = pairBaselineAndCandidateRuns(consumers, changes, release). Use identical inputs and independent consumer acceptance criteria, then classify each changed result.
4. Bind waves = sequenceMigrationByRisk(replay, consumers, release). Preserve the supported window for pinned customers. Unowned high-risk integrations block their wave.
5. Bind rollback = testRollbackDataCompatibility(waves, release). Require explicit handling for candidate writes the baseline cannot represent.
6. Bind communications = bindReleaseCommunications(release, changes, waves). Early documentation must remain clearly tied to the candidate version, not presented as current guidance.
7. Return { releaseId: release.releaseId, changes, consumers, replay, waves, rollback, communications, readiness: decideReleaseReadiness(changes, replay, waves, rollback, communications, release), commitAllowed: false }.
