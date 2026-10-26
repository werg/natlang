---
args: { prepared: PreparedMeasurementFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsMeasurementFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Compare metric definitions, event schemas, instrumentation notes, and missingness reports. Identify whether treatment changed the measurement process itself; formulate exact diagnostic queries instead of assuming a numerical difference reflects changed customer behavior.
2. Return field-level atoms for the actual MeasurementFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
