---
args: {"decisions": "LedgerDecision[]", "premises": "PremiseLink[]", "changes": "PremiseChange[]", "pending": "ReviewQuestion[]"}
returns: "ReviewQuestion[]"
---
Group review work only when decision owner, unresolved premise, evidence requirement and authority coincide. Preserve separate independent issues and cycle membership. For each group formulate alternatives an owner can decide, with surviving scope and the precise changed condition. Bind expectedRevisions to the current decisions. Reuse an existing identical open question rather than creating another notification. A new material change makes the old question stale and creates a replacement with that change visible.
