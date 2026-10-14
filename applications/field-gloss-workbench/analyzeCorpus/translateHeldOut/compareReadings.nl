---
description: Group semantically equivalent target readings without losing different analyses.
args:
  target: TargetRequest
  readings: MeaningReading[]
  rules: SourceClause[]
returns: TranslationAlternatives
---
1. Bind actual per-hypothesis readings for this original target. Check distinct actual hypothesis IDs and preserve raw child results before grouping.
2. Interpret source output equivalence rules: original entity/event IDs, polarity, topic and asserted temporal/future contribution determine semantic equivalence; free reason/wording and generated IDs do not.
3. Compare actual structured semantic fields across readings. A null unresolved field is not equivalent to a decided false value; two free temporal paraphrases can be equivalent when the same source-defined contribution is established.
4. Build each actual equivalence group with all supporting hypothesis IDs and stable reading identities derived from target/hypothesis pairs. Do not merge the underlying global hypotheses merely because this target fails to discriminate them.
5. Inventory fields whose decided meanings differ consequentially across completed readings. Keep prospective expectation differences even when event polarity and participants agree.
6. Preserve incomplete child readings in unresolved with actual hypothesis IDs; neither select the first completed survivor nor omit a difficult alternative.
7. Return raw actual readings, source-grounded equivalence groups, ambiguity fields and blockers. These are the alternatives consumed by both final audit and authorized note repair.
