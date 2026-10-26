---
args: { prior: QualitativeStudyCheckpoint | null, proposal: QualitativeStudyProposal, impact: StudyRevisionImpact }
returns: QualitativeStudyCheckpoint
---
For a first run, create revision 1 with the case, corpus edition, frozen selected sample, codebook versions, coder comparisons, and pending privacy reviews. For a repeated identical proposal, return the checkpoint unchanged. For a changed corpus, sample or codebook, append the proposal and changed codebook editions; retain prior comparison records, mark only impact.staleCodingIds invalid, and keep the original frozen sample with its original version. Do not claim that an analysis was exported or that participant consent was obtained unless a matching receipt is already recorded. Advance once for a real change; never advance for a duplicate replay.
