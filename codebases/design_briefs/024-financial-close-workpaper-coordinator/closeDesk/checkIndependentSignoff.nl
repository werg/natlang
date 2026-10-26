---
args: { request: CloseCycleRequest; scope: CloseAssertionScope; conclusion: CloseConclusionDraft; prior: WorkpaperSignoff[] }
returns: CloseSignoffReview
---
Compare preparer and reviewer identities and inspect their supplied role evidence. Require a complete conclusion with no material residual, a current source edition for every scoped account and an exact conclusion/source binding. A prior signature applies only to its recorded binding; invalidate it only when the changed source dependency intersects this scope. Return eligibility and exact missing prerequisites; never emit a signature.