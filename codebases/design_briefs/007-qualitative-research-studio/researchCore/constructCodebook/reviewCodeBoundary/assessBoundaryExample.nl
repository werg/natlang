---
args: { candidate: CodeBoundary, passage: EvidencePage }
returns: CodeExampleJudgment
---
Read this passage against the candidate's inclusion and exclusion rules. Identify what it explicitly describes before inferring a mechanism. Classify it as an inclusion example only when the rule fits its cited context; classify it as an exclusion when a nearby but distinct experience is ruled out; classify it as a counterexample when the candidate boundary fails to separate the passage; otherwise keep it ambiguous. Return the passage ID, observed meaning, exact boundary reason, and source evidence IDs. Do not invent a participant interpretation.
