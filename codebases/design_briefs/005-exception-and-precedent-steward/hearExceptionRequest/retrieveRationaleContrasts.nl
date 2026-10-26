---
args: {"request": "ExceptionCase", "boundaries": "WaiverBoundary[]", "hearing": "ExceptionHearing"}
returns: "RationaleAnalogy[]"
---
Retrieve by the reason for deviation and facts that made it acceptable, including a negative/expired precedent slice. For each candidate bind analogy = comparePrecedentRationale(request, boundary, hearing.caseHistory). Preserve disanalogies that change the rationale. If source access is restricted, provide an opaque marker and request authorized review; never leak private rationale to the current reviewer. No-precedent clauses bar treating the grant as binding precedent even if facts look similar.
