---
args: { code: CodeBoundary, mechanism: string, passage: EvidencePage }
returns: { role: 'support'|'contradict'|'narrow'|'irrelevant'; reason: string; evidenceIds: string[]; revisedScope: string|null }
---
Compare the passage with the stated mechanism and the code boundary's exclusion and counterexample examples. Call it support only when the passage directly bears on the proposed mechanism; call it contradict when it supplies a materially contrary case; call it narrow when only a condition or subgroup remains supported; otherwise call it irrelevant. Return the actual reason, source IDs, and a revised scope only when the evidence warrants narrowing.
