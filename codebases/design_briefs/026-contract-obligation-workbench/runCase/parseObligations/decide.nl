---
args: { prepared: PreparedContractObligation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsContractObligation
---
1. Apply this domain algorithm to each relevant evidence unit: Extract atomic obligations with actor, beneficiary, trigger, deadline, and exceptions. Separate descriptions and aspirations from operative provisions, retain cross-references, and preserve unknown roles rather than assigning responsibility from nearby names.
2. Return field-level atoms for the actual ContractObligation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
