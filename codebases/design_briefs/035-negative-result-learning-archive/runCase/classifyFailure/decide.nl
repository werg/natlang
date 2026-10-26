---
args: { prepared: PreparedNegativeFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsNegativeFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Determine whether evidence supports a negative result, an invalid measurement, an unfinished procedure, or genuine uncertainty. Preserve the distinction even when all cases were informally described as failures and avoid stronger conclusions than the attempt warrants.
2. Return field-level atoms for the actual NegativeFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
