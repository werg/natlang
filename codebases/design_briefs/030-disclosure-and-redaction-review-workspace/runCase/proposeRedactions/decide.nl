---
args: { prepared: PreparedRedactionProposal, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRedactionProposal
---
1. Apply this domain algorithm to each relevant evidence unit: Apply supplied rules to personal, confidential, privileged, or third-party material. Propose the smallest redaction that protects the restricted information while preserving the permitted claim; identify cases requiring qualified interpretation rather than making final legal judgments.
2. Return field-level atoms for the actual RedactionProposal record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
