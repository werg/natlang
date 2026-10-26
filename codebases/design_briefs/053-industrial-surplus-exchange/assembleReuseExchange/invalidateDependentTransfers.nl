---
args: {"profiles": "MaterialProfile[]", "legs": "ExchangeLeg[]"}
returns: "string[]"
---
Seed the affected set with withdrawn offers and newly prohibited transfers. Traverse the finite prerequisite graph until every dependent leg is accounted for, carrying visited IDs so cycles become explicit coordination blockers. Invalidate unconfirmed promises before other legs are confirmed. Preserve actual custody, received quantities and externally unknown cancellations; a local withdrawal cannot make material already in transit available again.
