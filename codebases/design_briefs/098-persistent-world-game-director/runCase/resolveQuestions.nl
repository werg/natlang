---
description: resolveQuestions for Persistent World Game Director.
args: { blockers: string[]; replies: string[] }
returns: string[]
---
Resolve blockers only from replies that address their named missing fact and come from an authorized respondent. Deduplicate repeated questions; leave unrelated blockers open.
