---
args: { passage: InterviewPassage, code: CodeBoundary, evidence: EvidencePage[] }
returns: CoderDisagreementReview
---
Bind sources = qualifyCoderEvidence(passage, code, evidence). Compare the actual independent labels and cited context, not coder seniority. Classify disagreement as boundary, translation, interpretive, or insufficient-record from those sources. Preserve a credible minority reading verbatim in meaning, attach its evidence IDs, and ask a narrow resolution question only when evidence could distinguish the readings. Return the exact passage, code/version, labels, classification, minority reading, evidence and optional question.
