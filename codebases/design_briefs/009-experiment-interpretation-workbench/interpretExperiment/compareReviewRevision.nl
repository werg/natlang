---
args: { prior: ExperimentReviewCheckpoint | null, current: ExperimentInterpretation }
returns: ExperimentChangeImpact
---
Compare analysis IDs and their exact design version, outcome window, measurement schema, assignment unit, approval reference and search-history binding with the saved interpretation. Invalidate only result claims consuming a changed or newly noncomparable measurement. Preserve results from unaffected windows and do not infer that an absent output was recomputed. Reopen follow-up work when the observation period or approved analysis changed; preserve the prior interpretation as historical evidence.
