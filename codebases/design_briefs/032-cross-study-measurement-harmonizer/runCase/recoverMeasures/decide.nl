---
args: { prepared: PreparedMeasureDefinition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsMeasureDefinition
---
1. Apply this domain algorithm to each relevant evidence unit: Read dictionaries, questionnaires, protocols, and usage notes together to recover what each variable measures. Distinguish similar labels from equivalent constructs and record timing, respondent, and collection-method differences that could change interpretation.
2. Return field-level atoms for the actual MeasureDefinition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
