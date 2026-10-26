---
args: { prepared: PreparedInquiryFrame, goal: GoalContract, policy: PolicySelection }
returns: DecisionsInquiryFrame
---
1. Apply this domain algorithm to each relevant evidence unit: Interpret the decision that research should inform and decompose it into answerable questions. Identify plausible rival explanations and what the available corpus cannot establish; avoid converting an exploratory question into a predetermined conclusion.
2. Return field-level atoms for the actual InquiryFrame record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
