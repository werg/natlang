# Pop Luna s52059 rejection review (2026-10-14)

The batch closed 118/118 cases, 107 accepted and 11 rejected by the collection
oracle. The four rejected `knowledge_research` cases all have `span` oracles
with `needs_review: true`; they are **not** established model failures:

| Case suffix | Gold | Teacher answer | Disposition |
| --- | --- | --- | --- |
| `e73d9cc2591f5721b1eb82a863374aae:v0:hinted` | `two-time` | `2 times` | Correct numeric equivalence in the supporting Lin Dan article; source-bound review needed. |
| `e73d9cc2591f5721b1eb82a863374aae:v0` | `two-time` | `twice` | Correct numeric equivalence; same source group as above. |
| `14d45d92a14c7c836ca94abc4781b99d:v0:hinted` | `Clay Regazzoni` | `Clay Regazzoni, a Swiss driver.` | Correct entity but adds a descriptor to an answer-only request; hold pending format review. |
| `8510ca8d173846cbac3d7ab02d0d9b56:v0:hinted` | `Annihilation` | Answer names *Annihilation* and cites the award and Alex Garland film evidence. | Correct entity but substantial extra explanation violates the answer-only request; do not admit as an SFT target. |

The corpus admission helper previously converted any unreviewed span answer
with token F1 at least 0.3 into an accepted row. That threshold can accept a
wrong named entity sharing one token with the gold answer. Pop removed the
automatic override in `ts-host/scripts/inline-curriculum/admit.mjs`; the
oracle's `needs_review` now remains a hold until independent source-grounded
review. The focused CLI test and full main check passed (Pop ledger unit
`natlang-pop-full-main-span-admission-v1`, exit 0; Python, Neuralese, GPU,
TypeScript, conformance). An audit of 48
`runs/*admitted*.jsonl` files (1.81 GB) and nine similarly named corpus files
found no stored `unjudged free text, span F1` marker; this is a bounded audit,
not a proof for every prior representation.

One `dataset_workbench` rejection is a source-label false negative:
`inline-curriculum:s52059:dataset_workbench:dataset10:a` routes M1 to the
default from Banking77 label `compromised_card`, but the visible text says
“I don't recognize these transactions,” which satisfies the generated
unrecognized-payment rule. Keep it out of failure/DPO training. The other
four dataset-workbench rejections inspected were semantic misses, not evidence
of a general runtime failure: Intel dividends classified as science rather
than business, film praise classified as a pan, a strongly negative review
classified as praise, and an iTunes pricing news item missed by its news rule.

Next: implement a reproducible, exact-identity post-collection review for
the two concise numeric equivalents (both share one source group). Pin
source/question/gold/visible supporting text and exact teacher string. Admit
only a reviewed derivative, never a fuzzy global comparator. Keep the longer
answers held or use them only after a separate format-quality decision.
Check all new span rejections through the same review path before treating
them as model negatives. For dataset workbench, build an ambiguity review or
filter for source-label overlap before expanding its exact-oracle share.

## Follow-up from Luna s52060

At least three further `dataset_workbench` rejections expose the limits of
using a source classification label as an exact answer to a newly worded
semantic question. AG News labels a story about former NBA player Calvin
Murphy's sexual-assault trial as `Sports`, but the generated question asks
whether it is *mainly about sport*. Another story about a soccer club's
murder charge is labeled `World`, while the question asks whether it is mainly
about international politics or world events. SST-2 labels a morally mixed
film comment `positive`, while the teacher routed it as a negative review.
These are unresolved label/criterion disagreements, not reliable model
negatives. Do not turn them into DPO failures. Review the item-level labels
against each generated criterion before widening this family's exact-oracle
share; a family-level source label alone is insufficient evidence.

The new HotpotQA rejection has gold `score of 37–24` and teacher answer
`Super Bowl XXVI: 37–24.` The score is present but the answer is longer than
the answer-only request. The revised admission path correctly leaves it for
source and format review. The other new failures were a route-planning
hazard count of one rather than zero and an edit-stream title that omitted
the required double-angle-bracket format; neither points to a runtime fault.
