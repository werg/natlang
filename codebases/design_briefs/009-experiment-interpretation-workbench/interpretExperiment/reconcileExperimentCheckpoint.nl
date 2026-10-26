---
args: { prior: ExperimentReviewCheckpoint | null, current: ExperimentInterpretation, impact: ExperimentChangeImpact }
returns: ExperimentReviewCheckpoint
---
For a first review create revision 1 with the case, review version, interpretation edition, exact output identities and unresolved approval references. If the same review version and output identities were already interpreted, return the checkpoint unchanged. Otherwise append a new edition; preserve every earlier measurement finding and interpretation, add only the newly invalidated analysis IDs to stale history, and retain recommendation lineage. Never mark approval, persistence or data collection complete from a proposal. Advance once for a new version or changed exact output and never for duplicate replay.
