---
args: { questions: string[], input: CaseInput }
returns: QuestionBatch
---
Deduplicate blockers, name an authorized respondent, and state acceptable evidence. A general acknowledgment does not answer every blocker.
Return the typed result, preserving missing facts as missing and citing evidence IDs where a decision depends on evidence.
