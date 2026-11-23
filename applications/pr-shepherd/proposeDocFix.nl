---
description: Draft a documentation update addressing grounded PR findings and any verifier problems.
args:
  findings: Finding[]
  targetPath: string
  existingDoc: string
  problems: string[]
returns: DocFixDraft
---
Update the documentation file at targetPath starting from existingDoc so that it clearly documents the behaviors, constraints, or migration notes raised by findings.
Preserve existing headings and unrelated sections from existingDoc.
If problems is non-empty, it lists exact verifier failures from an earlier draft; fix every reported problem cleanly.
Return the updated file path, complete new file content, and a one-sentence summary of the update.
