---
args: {"atlas": "IntegrationAtlas", "change": "CutoverChange"}
returns: "CutoverReview"
---
Compare expectedAtlasRevision and sponsor authority with the exact current atlas. If stale or unsupported, leave it unchanged and return the missing binding. For the selected unexecuted wave keys change plannedExecutionAt only, recording the reason and approval evidence. Preserve effectiveDecisionAt, historical aliases, already effective obligations and actual completed receipts. Reassess time-sensitive prerequisites against the new planned date and hold affected waves when their authority or integrity receipt is stale. Never rewrite historical decisions to make a postponed cutover appear on time.
