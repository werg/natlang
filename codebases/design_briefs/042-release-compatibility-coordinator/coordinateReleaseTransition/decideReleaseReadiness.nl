---
args: { changes: ReleaseBehaviorDelta[], replay: PairedCompatibilityReplay[], waves: MigrationWavePlan, rollback: DataRollbackPlan, communications: ReleaseCommunications, release: CompatibilityReleaseRequest }
returns: ReleaseReadinessDecision
---
Block the readiness claim when revisions disagree, a breaking replay lacks a migration, an integration remains unowned, a rollback can lose data, or communication describes a different version. Mark review where consumer acceptance is unresolved. Return ready for release-owner approval only when every blocking compatibility and communication condition is closed.
