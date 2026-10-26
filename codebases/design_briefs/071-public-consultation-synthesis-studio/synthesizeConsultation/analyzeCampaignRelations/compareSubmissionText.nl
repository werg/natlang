---
description: Compare template overlap and independent contribution at submission and argument scope.
args: { docket: ConsultationDocket; arguments: ScopedArgument[] }
returns: SubmissionRelation[]
---
Compare normalized text only for relation discovery; preserve original submission content unchanged. Use exact source identity for duplicate ingestion. Classify template relations only when shared language is evidenced, and record every argument or reason contributed beyond the template. If text is similar but independent authorship cannot be determined, leave the relation unknown rather than merge counts.
