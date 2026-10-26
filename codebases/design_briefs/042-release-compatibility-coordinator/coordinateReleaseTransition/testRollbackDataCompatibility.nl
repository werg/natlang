---
args: { waves: MigrationWavePlan, release: CompatibilityReleaseRequest }
returns: DataRollbackPlan
---
Compare every field and write introduced by the candidate with the baseline's readable representation. If rollback would discard candidate-era data, define a reversible transform or a reconciliation pause before authorizing the wave. Require an explicit owner decision for irreversible loss; a code rollback alone is not data compatibility.
