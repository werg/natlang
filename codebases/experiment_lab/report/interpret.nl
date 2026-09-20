---
description: Explain a finite experiment while distinguishing exact measurements from semantic judgments.
args:
  question: Text
  plan: Plan
  metrics: Metric[]
  trials: Trial[]
returns: Analysis
---
Interpret the exact `args/metrics` for `args/question`. State denominators and failed or
missing trials. Repeated value digests measure repeatability for those paired completed
runs only; they do not establish semantic correctness or a merge law. A quality label of
"pending" means nobody has admitted the semantic result. Put unsupported conclusions in
`unknowns` and suggest concrete `followups`. Do not replace the exact counts with guesses.
