# failure-explainer: advisory explanations of failures

Plans/FAILURE_EXPLANATION_PROGRAM.md (review items P5 and P11). Three natural-language functions read crisp facts and
explain; crisp code checks every answer. The output is advisory: it goes to separate files labelled with the
explainer's hash, and nothing the pipeline acts on reads it (audit counts and tags, `classifyAdmissionReason`,
`dpoHoldReasons`, gates, admission, the training mix).

```
explainFailure.nl       one student-projection failure the audit left unclassified: an existing tag, or a proposed new one
triageRejections.nl     admission reasons no rule classifies: category, action and a prefix for each
explainGateFailure.nl   a failed gate: where the failures concentrate, candidate causes, next checks
index.ts                crisp side: checks against inputs, bucket, rule verifier, explainer identity, advisory writer
refinements.ts          crisp checkers of the Is<...> result types
```

Entry points (all run after the crisp tool or gate has finished):

- `python3 scripts/audit_student_projection_failures.py DIR --out R.json --explain-input CARDS.json`, then
  `node ts-host/scripts/explain-advisory.mjs failures --cards CARDS.json --server URL --model NAME`
- `node ts-host/scripts/explain-advisory.mjs rejections --ledger LEDGER.jsonl ...`
- `python3 scripts/explain_gate.py REPORT.json [--explain ...]` (computes the facts with the gate's own arithmetic and
  fails when its decision differs from the report's), then `explain-advisory.mjs gate`.

A proposed rule is accepted only by a person, as a code change with a test. `checkProposedRule` verifies a proposal
against every reason the classifier has recorded: the prefix matches its reason and covers no reason already classified
into another category.
