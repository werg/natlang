---
args: { prepared: PreparedCounteroffer, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCounteroffer
---
1. Apply this domain algorithm to each relevant evidence unit: Draft conditional counteroffers that exchange permitted concessions for improvements in the whole package. Keep supplier-confidential information isolated and make dependencies explicit so acceptance of one term cannot accidentally commit the buyer to an unapproved package.
2. Return field-level atoms for the actual Counteroffer record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
