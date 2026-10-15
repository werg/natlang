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
remain. LFM/Mellum twin rendering and protected union inclusion are pending.
