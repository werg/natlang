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
| Logical source weighting | Initial launch retains exact CE `1234d1f1…`. Physical re-chunking must not change the objective. The measured old coding share is excessive; the next declared family-scale transition uses the source-bound own-output census, preserving historical objective custody. |
| Source diversity | Cohort-head packing through `be63cc2bf`, with monotone physical cursors independent of fairness flags in `2723c47c5`; preserve every required visit and protected split. |
| Complete targets and current data | Initial 591,181-row source `9d93141d…`, including verified prompt variants and corrected communication targets. Use transfer v2 `fb4be9d9…`; v1 omitted nine logical evidence paths. Full and foundation named roles move together. |
| Correct native graph computation | BF16 convolution gather-boundary fix through `cc3551dbd`; retain one-pass producer/consumer gradients and hybrid-state handling. Cached equivalence is distinct from behavior improvement. |
| Actual tool contract | Complete XML tool parsing/schema/custody, full explanatory early-RL prompt, correct tokenizer/template and serving settings. Preserve exact runtime and actor identities. |
| Early on-policy learning | Alternate useful supervised work with real-outcome whole-graph JOLT/RL. Initial approved RL rates: backbone `1e-6`, interface/learned rows `1e-4`; regenerated recipe and initializer must agree. Do not let an older recipe silently restore `2e-5/.001`. |
| Matching policy precision | `1a61d7334`, retained in accumulation merge: BF16 actor and policy objective agree. Preserve QAT declaration and refresh calibration on returning to SFT after policy updates. |
| Behavioral and update monitoring | Measure held task execution, actual idiom use, reply learning, weight-change coherence, sparse-row updates, numerical health and useful throughput. Lower loss alone does not qualify the run; base-relative KL remains telemetry. |
| Recovery and prompt adaptation | Full optimizer/RNG/exposure exports independently readable without running sandboxes; verify actual objects and owner GET separately. Preserve live dataset append/reload and whole-recipe scheduling. |

## Active follow-up, with explicit reasons

- Big2-001/002: 226,019 rows have archive-backed custody, no indexed ID or
  complete graph-content collisions against current train/held data. The 817,200-row
  successor is sealed; rebase its complete declared CE and append proof onto
  the actual distributed recipe before adoption. The earlier incomplete 519-leaf
  CE is invalid for the 521-leaf successor. Prepared rows are not trained exposure.
- Historical DPO pairs: retain useful text and provenance, but re-encode for
  LFM3 and verify labels, protection, oracle custody and complete-graph capacity.
  LFM2.5 tokenized artifacts are not directly ready LFM3 training data.
- Long communication chains remain an admission/capacity obligation. Neither
  one-node SFT nor prefix counts establishes complete graph exposure. The owner's
  LFM3 direction explicitly removes cross-model stages and Mellum dependence.
- Kahan compensation is implemented. Initial distributed controls explicitly
  retain stochastic rounding while the real transition and fragment-aware outer
  residual handling are integrated and qualified for the assigned host memory.
  Do not silently change the optimizer algorithm under a recorded recipe.
- Full teacher prompt variants and the authenticated four-family classification
  are being sealed by the Claude process. Admit them through the same live
  source continuation, preserving useful header variants and protected worlds.
- The initial eight requalified RL cases are usable initial supply, not the
  complete curriculum. Continue expanding verified tasks and full graph capture.

## Current verification

Root integrated diversity/resident/JOLT checks passed 148 tests before the
accumulation merge. The subsequent combined accumulation, objective, initializer,
diversity, publication and precision checks passed 54 tests. Claude stopped its
local validation as redundant; termination interrupted its final HDD checkpoint.
That partial file is not a verified recovery state or capability result.

Final recipe forwarding, initial RL collection and fresh-parent custody checks
passed8 tests, followed by the quick checks on `bc319e606`. The full suite is
running frozen `bc319e606` plus the separately recorded four-file local overlay
under ledger `natlang-check-fresh-bc319-1029`, with scratch on the external disk.
That older full run exposed fixture-path and HDD timing failures; 28 targeted
HTTP/recovery/replica and assembler tests pass on current main with internal
temporary storage. Failed child cleanup and fixture path custody were repaired.
The full run remains incomplete; do not describe it as green.

Peer4's separate fresh-base JOLT diagnostic completed one actual update. All
three task executions succeeded, with zero outcome variance and no communication
edges: this qualifies neither outcome lift nor learned Neuralese communication.
Its checkpoint is preserved separately and is not transplanted into production.

Peer1's fresh SFT earned step202, then drained with optimizer/RNG/exposure intact.
The four-worker first outer session preserves this descendant on rank0 and uses
the exact common step2 seed `51bacc47…` on ranks1–3. The seed has independently
verified private owner readback; step202's newer independent export is tracked
separately. A retained local full recovery copy still requires verification.

All four hosts verified the 585-file b6c runtime, followed by the exact b0a mapped
source-eviction correction. The sealed v2 controls bind disjoint shards,
accumulation16, local_steps32, quorum3, coordinator grace30, BF16 wire and ten
average rounds before Nesterov. f3 hosts primary coordination and b166 the
replica, alongside their learners. Full coordinator initialization started at
PID78260; initialization is not yet evidence of four advancing learners or an
outer merge. Verify both before reporting distributed training active.

The cloud step33 evaluator was stopped to release rank3. Its partial unmatched
results do not establish fresh-run regression or improvement. Run matched
development evaluation on DGX or short bursts, preserving protected data.

Operational manifests and source inventory are under
`/home/werg/data/dgxops/claude-recovery-20261028/data/`. Incomplete manifests
remain explicitly non-launchable; update them with observed execution evidence,
not optimistic status labels. Missing unrelated future data must not block useful
training on already verified sources.
