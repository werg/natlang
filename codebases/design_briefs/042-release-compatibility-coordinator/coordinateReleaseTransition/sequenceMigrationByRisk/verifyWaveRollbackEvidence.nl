---
args: { wave: CoordinatedChangeWave, rollback: DataRollbackPlan }
returns: WaveRollbackQualification
---
For this wave, locate the exact rollback rehearsal, data reconciliation result, and responsible operator. Fail when any expected candidate-era write cannot be read or reconciled after rollback. A planned procedure without an observed rehearsal remains unqualified.
