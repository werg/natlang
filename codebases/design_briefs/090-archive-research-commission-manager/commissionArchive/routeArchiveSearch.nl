---
description: Route question-specific archive searches and preserve exact coverage states.
args: { request: CommissionRequest; sources: ArchiveSource[]; questions:string[]; prior:SearchObservation[] }
returns: SearchObservation[]
---
For each in-scope collection and question, retain prior observations and enumerate relevant query variants. Carry explicit access state. A closed or restricted collection is never called searched. Distinguish unsearched from exhausted and include the reason a collection can or cannot answer the question. Only supplied observations count as hits.
