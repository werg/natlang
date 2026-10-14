---
description: Compare exact candidate cost vectors under the source objective.
args:
  candidates: CandidateValue[]
  objective: string
  witnesses: Witness[]
returns: CandidateComparison
---
1. Read objective as an ordered comparison contract. Identify the requested completion condition,
   numeric priorities, direction of preference and final tie rule. Link each interpreted priority to
   the supplied witnesses. If competing interpretations can select different candidates, retain that
   ambiguity in reason and return null selection; iteration order is not an objective.
2. Inventory the actual CandidateValue experiment IDs and their answer branches. Distinguish a
   complete policy from a cheap partial policy: every positive-probability answer must have a retained
   child continuation. Interpret complete using the upstream goal, which may resolve a mechanism
   while leaving within-mechanism genotype uncertainty. Preserve that distinction in the comparison.
3. Separate candidates explicitly marked incomplete. Keep their experiment IDs and concrete missing
   branch reasons in the explanation. Numeric placeholder fields on incomplete candidates do not
   enter the ordering. If no complete candidate remains, return null with that obstruction.
4. Initialize the contender collection with all actual complete candidates. Visit each interpreted
   numeric priority in source order. At one priority compare the corresponding actual Fraction fields
   with exact cross-products and retain all candidates equal to the best value at that priority.
5. Preserve the reason for each elimination as the first priority at which that candidate loses.
   A later advantage cannot override an earlier source priority. In particular, distinguish expected
   resource use across possible answers from protection against the most expensive answer branch.
6. After the last numeric priority, retain the actual contender IDs as the numerical tie group. With
   a unique contender there is no unresolved action choice; tiedExperimentIds may be empty or contain
   that sole winner. With several contenders, interpret the
   source tie convention: apply an established identity ordering, allow any tied witness when the
   objective permits it, or return null when the source requires a choice but supplies no basis.
7. Select only an experimentId already present in that final contender group. Return that ID,
   tiedExperimentIds and a concise witness-backed account of completion, the decisive priority and
   any final tie resolution. tiedExperimentIds describes exact numerical ties, not merely similar costs.
8. Preserve the supplied candidates unchanged. The caller binds expectedCost and worstCost from the
   selected actual CandidateValue; this function owns comparison meaning and choice, not prediction,
   branch probabilities or a replacement policy. A missing source priority remains visible uncertainty.
