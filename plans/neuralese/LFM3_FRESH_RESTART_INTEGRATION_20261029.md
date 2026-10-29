# Fresh LFM3 restart: inclusion and execution record

Owner requirement, 2026-10-29: include feasible beneficial changes in the actual
new run. A checklist entry is not deployment or capability evidence. Bind the
resolved launch command, source, objective, model, runtime and recovery artifacts
in the machine-readable restart manifest before treating them as active.

The suspect four-peer lineage is diagnostic only. Restart from instruction base
`LiquidAI/LFM3-1.7B` revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`,
with fresh native interface, optimizer and RNG, and zero earned exposure.
Do not restore suspect weights, optimizer moments or exposure as this run's state.

## Changes selected for inclusion

| Change | Implementation and execution requirement |
| --- | --- |
| Genuine fresh initialization | `fd344fdd3`: save real step-zero model/optimizer/RNG and outstanding SFT obligations; no fabricated trained-parent or drain receipt. Initialization is not capacity qualification. |
| Larger effective optimizer batches | `f7289bebc`: accumulate 16 packed micro-batches before one update; preserve whole graphs and count actual optimizer updates separately from row visits. Target memory qualification must retain accumulated gradients. |
| Reply-token-aware loss | `f7289bebc`: generate and bind an explicit token-mass objective. The earlier invocation-mass CE declaration is historical and does not become token-mass by using newer code. Keep reasoning at its declared lower share. |
| Logical source weighting | Logical mechanism map and bounded mean-one scales through `2b12a1579`; physical re-chunking must not change the objective. Recompute realized token and task contributions under token-mass, rather than reusing graph-weight percentages. |
| Source diversity | Cohort-head packing through `be63cc2bf`, with monotone physical cursors independent of fairness flags in `2723c47c5`; preserve every required visit and protected split. |
| Complete targets and current data | Initial 591,181-row source `9d93141d…`, including verified prompt variants and corrected communication targets. Use transfer v2 `fb4be9d9…`; v1 omitted nine logical evidence paths. Full and foundation named roles move together. |
| Correct native graph computation | BF16 convolution gather-boundary fix through `cc3551dbd`; retain one-pass producer/consumer gradients and hybrid-state handling. Cached equivalence is distinct from behavior improvement. |
| Actual tool contract | Complete XML tool parsing/schema/custody, full explanatory early-RL prompt, correct tokenizer/template and serving settings. Preserve exact runtime and actor identities. |
| Early on-policy learning | Alternate useful supervised work with real-outcome whole-graph JOLT/RL. Initial approved RL rates: backbone `1e-6`, interface/learned rows `1e-4`; regenerated recipe and initializer must agree. Do not let an older recipe silently restore `2e-5/.001`. |
| Matching policy precision | `1a61d7334`, retained in accumulation merge: BF16 actor and policy objective agree. Preserve QAT declaration and refresh calibration on returning to SFT after policy updates. |
| Behavioral and update monitoring | Measure held task execution, actual idiom use, reply learning, weight-change coherence, sparse-row updates, numerical health and useful throughput. Lower loss alone does not qualify the run; base-relative KL remains telemetry. |
| Recovery and prompt adaptation | Full optimizer/RNG/exposure exports independently readable without running sandboxes; verify actual objects and owner GET separately. Preserve live dataset append/reload and whole-recipe scheduling. |

## Active follow-up, with explicit reasons

- Big2-001/002: 226,019 candidate rows have archive-backed custody, no indexed
  ID collisions and measured source geometry. Graph-content dedup/admission is
  still running; append promptly once verified. They are not trained exposure.
- Historical DPO pairs: retain useful text and provenance, but re-encode for
  LFM3 and verify labels, protection, oracle custody and complete-graph capacity.
  LFM2.5 tokenized artifacts are not directly ready LFM3 training data.
- Long communication chains and cross-model stages remain recipe obligations.
  Neither one-node SFT nor prefix counts establishes complete graph exposure or
  cross-model readiness. Preserve actual producer/consumer exposure requirements.
- Kahan compensation for BF16 optimizer rounding is a follow-up if measured
  noise remains problematic after accumulation. It is not implemented or
  silently included; monitor the new run rather than assuming accumulation
  cures every cause of the regression.
- The initial eight requalified RL cases are usable initial supply, not the
  complete curriculum. Continue expanding verified tasks and full graph capture.

## Current verification

Root integrated diversity/resident/JOLT checks passed 148 tests before the
accumulation merge. The subsequent combined accumulation, objective, initializer,
diversity, publication and precision checks passed 54 tests. Local Claude fresh-base
validation continues under `claude-v4-validate-4`; do not duplicate it or treat
its checkpoints as production capability lift. A final runtime must include the
merged fixes above, with target capacity and actual learner start recorded.

Final recipe forwarding, initial RL collection and fresh-parent custody checks
passed8 tests, followed by the quick checks on `bc319e606`. The full suite is
running frozen `bc319e606` plus the separately recorded four-file local overlay
under ledger `natlang-check-fresh-bc319-1029`, with scratch on the external disk.
Peer4 saved and verified actual fresh state SHA `752f16c0…` with zero updates;
JOLT setup is in progress. Peer1 fresh controls are staging independently.
Neither initialization nor configuration is credited as completed training.

Operational manifests and source inventory are under
`/home/werg/data/dgxops/claude-recovery-20261028/data/`. Incomplete manifests
remain explicitly non-launchable; update them with observed execution evidence,
not optimistic status labels. Missing unrelated future data must not block useful
training on already verified sources.
