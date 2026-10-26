---
args: { request: CloseCycleRequest; workpapers: CloseWorkpaper[]; evidence: EvidencePage[] }
returns: CloseAssertionScope
---
Select only workpapers whose case, entity, fiscal period and requested accounts match. Bind each selected account to the exact ledger and report source references cited by the workpaper. If an account has no current source edition, keep it in missingAccounts and do not silently substitute a prior period. Carry the assertion and supplied materiality rule without expanding them. Return the pinned scope and its unresolved account or source gaps.