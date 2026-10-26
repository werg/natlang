---
args: {"premises": "PremiseLink[]", "prior": "DecisionLedger", "acts": "AttributedAct[]", "inbox": "DecisionInbox"}
returns: "PremiseChange[]"
---
For each actual correction, compare the old proposition with the new proposition in their common subject, scope and effective interval. Parse and compare the supplied times exactly; an unknown boundary leaves temporal applicability unresolved. A synonym, title rename or copied note changes neither premise nor independence. Record contradiction or narrowing only in the supported intersection, retain surviving decision scope outside it, and append both recorded and effective dates. A newer upload alone does not override an authorized earlier decision.
