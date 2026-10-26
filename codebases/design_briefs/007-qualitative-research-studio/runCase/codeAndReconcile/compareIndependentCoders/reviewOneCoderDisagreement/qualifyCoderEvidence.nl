---
args: { passage: InterviewPassage, code: CodeBoundary, evidence: EvidencePage[] }
returns: { coderLabels: string[], contextEvidenceIds: string[], translationEvidenceIds: string[], sourceConflictIds: string[] }
---
Locate the coder records that explicitly identify this passage and the supplied code version. Separate source-language passage evidence from reviewed translation and coder rationale. Return only the actual coder labels, context references, translation references and conflicting source identities; missing records stay absent rather than becoming a label.
