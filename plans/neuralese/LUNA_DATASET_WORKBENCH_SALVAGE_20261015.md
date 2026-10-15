# Held dataset-workbench salvage (2026-10-15)

Forty exact accepted Luna dataset-workbench parents from seeds 52067–52072 had
been held for semantic-label review. An independent item-level audit checked
their visible criterion, inputs, expected answer and observed result. Thirty
parents are semantically clear and ten remain held. Two held parents have
concrete label conflicts: an online-shopping payment decline was routed away
from the requested payment-failure class, and a cash/cheque deposit request was
routed away from the requested adding-money class. Other holds involve
underspecified scam, film-polarity or news-topic boundaries. The source-hash
bound audit is registered as `luna-dataset-workbench-held-audit-20261015-v1`
(manifest `b940ce5c`) and verified on DGX. It is review evidence, not training
data.

The thirty clear exact parents were rematerialized with the frozen collection
runtime: 841 native decisions, of which 769 passed per-decision approval and 72
were excluded. The curated source corpus
`luna-dataset-workbench-salvaged-20261015-v1` is registered with manifest
`4ce1e717` and verified on DGX. It has no duplicate decision IDs or source
groups against the original admitted seed52067–52072 corpora. Original seed
corpora remain immutable. The 69,763,284-byte candidate-turn materialization
was pruned after curation; the curated source and exact parent/audit evidence
remain. LFM/Mellum twin rendering is registered as
`luna-dataset-workbench-salvaged-render-20261015-v1` (manifest `c0207e02`),
verified on DGX: all 769 curated turns selected, zero unexpected omissions,
with 5% of programs sampled for full-prompt warmup. Protected union inclusion
remains the DGX owner's next admission step.

## Continuing audits

Seeds 52073–52078 each received an independent, source-hash-bound item review
before their accepted teacher parents were curated. The corrected audits
approved 11/12 dataset-workbench parents across 52073–74, 4/5 in 52075, 6/7
in 52076, 5/5 in 52077, and 4/5 in 52078. Held items cover ambiguous film
sentiment, card status, technology-news topics and the scope of “international
politics or world events.” Exact success alone was insufficient for admission.

The seed52077 v1 audit contained a false hold: it said M3's expected and
observed route was `cards`, while both exact source values say `trust`, which
fits the positive-film criterion. The corrected v2 audit, manifest `4a2d6788`,
checks exact source text, day, expected and observed values. Its corrected
source and LFM/Mellum twin render are DGX-verified as
`luna-held-shapes-s52077-curated-20261015-v2` and
`luna-held-shapes-s52077-curated-render-20261015-v2`: 85 parents and 744
approved native decisions, with no render omissions. The v1 registry entries,
manifests and redundant copies were retired. The remaining three whole-parent
holds are two variants of a contradictory HotpotQA adult/young-adult question
and one research trajectory that skipped required `iterateOn`.

Seed52078's first audit also falsely claimed a Bush–Kerry item was on day18;
the exact source says day4, before the task's day11 cutoff. Corrected audit v2
approves that parent and holds only the ambiguous world-events scan. The
corrected source is DGX-verified with 89 parents and 663 approved decisions;
its twin render is in progress. Protected union admission remains the DGX
owner's decision.
