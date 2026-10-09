---
description: Whether the inspected result meets what the request meant.
args:
  request: string
  source: SourceFacts
  plan: PlanFacts
  observed: Observed
  note: Untrusted<string>
returns: Assessment
---
Judge whether the rendered clip meets the user's meaning. The observed values are the evidence.

1. Restate what request asks in one phrase, using note when the request points to it.
2. Compare that with plan, source and the observed values.
3. When observed.visual_status is "contradicted", answer intent_met false.
4. Write explanation in one or two sentences that name the observed values it relies on, and any question the
   observed values leave open.
