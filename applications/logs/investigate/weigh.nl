---
description: Weigh the evidence found for one hypothesis.
args:
  hypothesis: Hypothesis
  found: Found[]
returns: Support
---
Decide what the evidence in found says about hypothesis. found holds the results of the searches planned for it; each
has the records (evidence) and the total number of matches.

1. Read each record's message and level.
2. stance: "supports" when the records show what the hypothesis claims, "contradicts" when they show it did not
   happen or something else explains the events, "neutral" when they say nothing either way or there are none.
3. evidence_ids: the IDs of the records that bear on the claim, from the evidence only.
4. note: one sentence on why.
