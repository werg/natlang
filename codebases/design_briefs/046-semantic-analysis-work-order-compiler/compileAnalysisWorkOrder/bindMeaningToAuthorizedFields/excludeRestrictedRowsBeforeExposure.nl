---
args: { accessMap: RecordAccessRule[], requester: ID | null }
returns: AuthorizedAnalysisUniverse
---
Remove records the requester cannot access before sending any row to a semantic classifier. If requester identity is absent, do not expose any record until an authorized identity is supplied. Preserve a count of excluded rows only when policy permits aggregate disclosure; otherwise return coverage as unknown. Never expose identifiers or values of restricted rows.
