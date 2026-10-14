# Dataset workbench rejection audit — 2026-10-14

The Pop Luna s52038 and s52039 shards completed 102 and 81 queue entries
respectively. Their `dataset_workbench` rows produced 5/6 and 2/6 answer
rejections; these are exact-answer failures, not collector infrastructure
failures. The rejected rows are excluded by ordinary admission.

The high rate is partly a source-label-to-task mismatch. Examples:

| Case | Source label | Task wording / visible fragment | Teacher decision |
| --- | --- | --- | --- |
| s52038 dataset14 M8 | SST2 positive | “you feeling like you've seen a movie instead of an endless trailer” | Routed as negative. The fragment lacks its original sentence opening. |
| s52038 dataset9 M1 | SST2 positive | “offers instead an unflinching and objective look at a decidedly perverse pathology” | Routed as negative. “Perverse pathology” makes sentiment hard to infer from this fragment. |
| s52038 dataset0 M7 | AG News Sports | “Help for Home Team” discusses mayoral funding for a stadium, neighborhoods and schools | Routed as non-sport under “mainly about sport.” The source topic and generated criterion differ. |
| s52039 dataset12 M6 | SST2 positive | “runs through a remarkable amount of material in the film's short 90 minutes” | Routed as non-positive. The fragment is compatible with praise or criticism. |
| s52038 dataset10 M5 | Banking77 activate_my_card | “I have a new card and need to activate it” | Misrouted to an unrecognised-payment team despite the text; this looks like a model mistake. |

The generator (`ts-host/scripts/inline-curriculum/dataset-workbench.mjs`) maps
dataset gold labels to more specific natural-language bucket predicates. Exact
gold membership is a sound oracle only when the *visible* snippet clearly
supports that predicate. Do not relabel a fragment merely to match one teacher.
Before increasing this family's generation share, obtain independent semantic
agreement on the generated bucket predicates and screen or rewrite the
ambiguous source examples. Keep clear, independently supported rows and the
other source families. A source change is model-facing and needs the usual
matched live measurement (about 48 samples per adopted variant). This audit
does not grant admission to any rejected row or discard the underlying
datasets.
