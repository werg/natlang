# Pop current work

Last reviewed: 2026-10-09, shared AR supervision correction. This page is a navigation aid; inspect live processes and receipts before acting. Historical evidence and course changes remain in [HANDOVER.md](HANDOVER.md) and [GENERATION_DECISIONS.md](GENERATION_DECISIONS.md).

## Ownership and coordination

- Pop owns execution here; DGX owns its execution. Canonical checkout on both: `/home/werg/natlang`. Do not launch from retired mirrors or change DGX jobs.
- Run `python3 scripts/coordination_inbox.py check --ack` at session start, before resource changes and each monitoring cycle. Use Git for code and the corpus registry/manifests plus `sync_training_corpora.py` for selected data artifacts.
- The user wants autonomous work and monitoring, including waiting/sleeping when jobs are in progress. Keep five authorized Luna slots supplied where appropriate. Do not report planned workers as actual provider activity.
- Current shared-work ownership: DGX is leading the shared objective/schedule and recipe-launcher consolidation; Pop owns the C++/WASM read-port transport for the reader adapter. Coordinate before editing shared renderer/window helpers. The current Pop map run is frozen and does not include later shared-runtime changes.

## GPU: full-depth autoregressive foundation continuation

Current state: the AR continuation and paired GPU diagnostics below have
completed. No GPU trainer is active while the next common-recipe continuation
is being prepared. The shared causal-prefix supervision and checkpoint tests
pass79 focused CPU tests. The next continuation
must use the immutable mapped best23936 pair, not the regressed final24960.
An abandoned closed repair's empty checkpoint-space reserve was released after
a privileged host-process check found no FD or job references; receipt is
`.coordination/closed-anchor-reserve-release-v1.json` (1,435,903,552 bytes).
Recheck available storage and atomic-save reservation before launch.

Course correction: sharing a repository did not prevent divergent recipes and
stale sketch assumptions. Pop incorrectly resumed sketch training after the
mapped-input decision. The sole active text implementation is now the shared
mapped-input trainer followed by its full-depth AR fixup. The shared
`causal_gold_prefix_mask` supervises the first differing decision and excludes
later gold targets from the generated history; gold control and held diagnostics
retain complete spans. This is a common objective change, not a Pop experiment
fork. DGX owns the broader text/trajectory objective and optimizer-handoff
consolidation; that work remains outstanding until its code and tests land.

The superseded sketch-consumer repair is closed and is artifact evidence only;
its best step3328 is not a qualified source. The preceding Pop job was the mapped-input continuation
`luna-map-refresh-step22272-v5` (container `98c449e7ef60`), launched from the
shared runtime at commit `d915133f695ced59fde70d71100735cc9ccd2ca7` using the
pinned image `sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de`.
Its declared recipe is
`training/neuralese/recipes/luna-map-text-4880-from-qualified-22272-20261009-v5.json`
(SHA `2043c442d5f740c183031203dc9a218424ee9ab6bc131b3f2843a1376cb66a84`).

The authoritative same-map source is the full checkpoint at step22272, SHA
`9d56437d775bf88bb4c156e129a73f43b60f914fb05bcb28fcf590340000aa8f`. Restore
logs confirm all four optimizer groups and the foundation schedule/RNG restored;
no fresh-optimizer fallback occurred. The source was previously qualified only
for its earlier data and gates; this changed corpus has a fresh baseline and
must earn new gates. Input is the adopted4880-row text file, 3176 train / 1704
heldout, SHA `b10da15de2510b55635b5aaa85301204e559669ac239968bc94e4393d08e0572`;
the admitted87 target-bound documents are already in it and must not be appended
again. Native records/pieces are source bindings, not extra text rows.

At the latest handoff, the mapped run was safely stopped by signaling its
ancestry-verified trainer child; emergency checkpoint step24843 is preserved.
Best step23936 is immutably pinned for AR exposure, checkpoint SHA
`adc6bfd59a5e0e67fa243270035f32da4688d25a73cdbb5a0da83276c11ecb1d`
and heads SHA `c73d46511451ded3e6d304bce489765e0c08be458c44fe47500e022bf6834f48`.
Its original strict .995 gate remains failed. Six saved evaluations pass a .99
CPU advisory with the other original clauses; this is not qualification.

The AR fixup v1 launch restored all four optimizer groups but failed before
updates: the old mapped metric was named `shallow`, the shared handler names it
`input_map`. Shared fix `cf582398` performs only that checked semantic rename;
every other schedule setting must match. Eight focused tests and a CPU replay
of the real source checkpoint preserve31 observations/history and RNG state.

AR fixup v2 completed normally at step24960 as `luna-ar-feedback-best23936-v2`, recipe
`training/neuralese/recipes/luna-ar-feedback-fixup-4880-best23936-20261009-v2.json`
(commit `6dcaac85`, SHA `e1a50f2941c121658ac5c0daf4fd57caaf193372426c594495860bec98b104bd`).
It pins shared code `cf582398` and the same Docker image. Logs confirm all
four optimizer groups and31 schedule observations restored. Final alignment
qualification is false; short/medium/long consumer agreement is
.783/.410/.307. The fixed projected AR control first diverges at target4
(crisp control remains exact for256 tokens). Final checkpoint and heads both
report step24960 with `heads_current: true`. Do not extend the failed run blindly:
use the matched diagnostics and investigate targets after generated-history
divergence before the next shared objective decision.
Driver/launch receipts are under
`/srv/storage/natlang-artifacts/neuralese-map-refresh-4880-ar-feedback-preflight-20261009-v2/`.
Failed v1 launch evidence remains preserved under its original path.
This binding uses16K total context,256 generated target positions, unchanged
gold targets,1024 additional updates, fresh .99/.05/.025 gates and two
consecutive passes. No transport or recurrence qualification is inherited.

### Matched AR evaluation

The six-window held-test crisp-versus-projected diagnostic completed as
`luna-ar-matched-source23936-v2`. Its source pair is the frozen step23936 checkpoint/heads; the optimizer
checkpoint is hash-checked only, while inference loads the paired heads export.
Selection and source pins are in
`runs/neuralese-ar-matched-multistratum-23936-20261009-v1/window-selection-v5.json`
(SHA `9b186dbf99d326890cb1c9e694144a8b077425b3567fb6f96d4745fd6586642e`)
and the frozen-code-bound `source-recipe-v5.json` (SHA
`a3735ddfa6b825b0e85e886a30f7973e48b66d7e9bedff8b73247d1762f3b6ad`). The
118-file Python package snapshot is under
`/srv/storage/natlang-artifacts/ar-matched-eval-v4/frozen-runtime-v1/`; its
inventory receipt SHA is `5be037143b19e9064c0f643ab2dbe3710fe2b94fd46ec51a4faad79bf596164b`
and matches the reviewed source-recipe inventory exactly. Use this frozen
package through `PYTHONPATH`, not mutable checkout code. The pinned image is
`sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de`.

The first GPU launch failed before model load because the window manifest still
pinned source recipe v4 after its code-root relocation to v5. The new v5 window
manifest changes only the recipe pin and ID; its six windows remain identical.
Both launch receipts and failure logs are preserved. The prior CPU attempt also
remains a preserved failure with no result; inference now runs on the GPU using
the heads-only loader and never deserializes optimizer state. Keep the report diagnostic
only. It grants no foundation, runtime, or task qualification.

The DGX owner direction is to phase the raw-token-embedding projection anchor
out after warm-in using a declared decay, then adapt the read port to learned
Neuralese vectors with a full-rank residual read adapter. It is not a request to
hot-patch this frozen map job. Existing evaluation-only autoregressive controls
remain limited: a fixed one-window full-projection control survived 256/256
tokens, while strict multi-stratum alignment remains false. Treat those as
evidence for the next declared stage, not as broad runtime qualification.

### Shutdown and storage

The active immutable runtime predates the repaired recipe signal forwarding. If an early stop is necessary, signal its ancestry-verified trainer child and verify the emergency checkpoint; do not rely on Docker stopping the outer PID1. Normal completion is safe. Future launches should use shared signal fix18770099; actual CPU-only Docker routing proof is in `runs/neuralese-recipe-signal-forwarding-proof-20261009-v1/receipt.json`.

The stopped sketch-consumer repair is preserved as held artifact evidence, not an
active training route. Its earlier Docker shutdown failed to save 73 updates
(serialized step1920 versus final log step1993); preserve that failure receipt,
current/best checkpoints, and the earlier recurrence checkpoint separately.

Latest storage check: about9.3 GiB free on `/` and2.8 GiB free on `/srv/storage`, plus the active run's separately allocated1.4 GiB checkpoint reserve. Its current and best full checkpoints are distinct. Avoid redundant checkpoint/model copies. Historical Step5 process state and unused grouped Ling base weights were handled with verified local offload/duplicate eviction; private process/browser caches are not training corpora.

## Generation and review

### Five Luna workers

Current campaign: `runs/luna-v26-counterfactual-fivecase-20261009-v1/dispatch-approved-v1/`.
Root approved plan SHA `fdca2f69aff6aa00b9623db31f48a2727d375eaa99085e07bf61fd94d53e543d`;
five workers have observed provider activity. Five train-only counterfactuals
inherit V26 source groups, with zero new-world credit. Runtime snapshot
`f17354c5ee64009031b057e38b4742fd1a68dec62317e5a49f7c897c6e7710fc`
was built from current source through the shared isolated compiler, including
the crisp-string read fix. Its1140 file hashes were independently verified.
The old runtime proposal was replaced before launch. Check journals/claims for
live completion status; completed tasks remain held pending action review.

The separate four-case packet under
`runs/luna-conjunctive-eligibility-derived-20261009-v1/action-review-v1/`
has50 reviewed native actions approved by the standard per-action receipt
`root-selected-50-admission-v1.json` (SHA `87e30fd3a02f3acbb7b71564b3c186a0f088cfcbce8b6a8ed09bdc9dc17d03a9`).
Root replayed the exact captured-request audit. Missing-score suspicions cleared
through authenticated expanded provider inputs, not oracle metadata. Shared
current isolated rematerialization and shared text conversion are root-adopted
as50 native /46 text rows. Actor joins are verified on every included row.
Full-current-union hash/collision/split checks and exact tokenizer/suffix decoding
passed. Published corpus ID: `luna-v17-conjunctive-eligibility-50-native-46-actor-text-20261009-v1`.
Four text omissions,179 other actions and four authored roots remain held. No recurrence/whole-trace/learned-vector credit.
Shared diagnostic fix96f9c697 clarifies `nl.with` result/capture generics and
`decide`'s callable contract; future snapshots include it, active frozen runs
retain their recorded implementation.

- The recent held packet has been reduced from199 proposals to22 after two definite visibility holds. The22 selected native actions and19 derived text documents were independently replayed and explicitly adopted;177 other actions remain held. Task acceptance does not grant whole-trajectory or recurrence admission.
- Earlier wave3 plan and its source/runtime proofs remain preserved under this campaign. Treat them as historical evidence; do not use their previous “latest wave” wording as current status.
- Review every observed action and skill disclosure before admission. No global v6 activation, new-world credit or blanket task-level admission.

### Step5 Preview Free

`runs/step5-preview-free-clinic-row0-corrected-20261009-v1` completed the correct Clinic source0/scenario2 and passed its oracle. It has18 actions,15 children and6 iterations. Owned idle bridge stopped; captures sealed. One malformed `return_result({code:...})` attempt is held separately; subsequent valid typed output is independently reviewed. Current shared adapters produced17 reviewed native actions and17 exact-serving-boundary text documents; root native and text replays are byte-identical. One action is an authored static root and16 are model samples; actor provenance is now mandatory in shared conversion. Token-suffix audit passed and root composition adopted17native/17text; curated publication/sync are verified on both machines; no recurrence or hidden-state admission. Step5 Microgrid terminated on repeated endpoint-unavailable503 errors, with captures sealed under `runs/step5-microgrid-concurrent-luna-wave2-20261009-v1`. No paid fallback or text-distillation flag.

### Failure repairs and skills

- Exact-context source audit confirms three genuine positive-eligibility false judgments (BIR-2C, RWA-1C, ESR-2A), with the relevant facts visible and faithful host captures. The failure boundaries were respectively a false leaf answer, an added identity/proof requirement in the delegated criterion, and a stale “eligibility unknown” premise passed to a child. The original three repair rows remain held.
- The shared `build-preference-pairs.mjs --source-derived-repairs PROPOSAL.json` adapter now accepts the exact-context v5 causal-boundary proposal. It joins source row, request hash, raw response hash, terminal tool-call ID, trace action and offered schema; it checks chosen eval syntax and names against declarations in the captured opening scope only, not the full persistent eval history. All three candidate pairs stay in the held sidecar pending per-item root preference admission; ordinary preference output remains empty. The chosen side preserves the eligibility delegation, and carries no task-success, runtime-equivalence or hidden-state claim.
- The audit's generic skill recommendations are to evaluate each stated conjunction against supplied facts, use later source-bound evidence instead of retaining a superseded unknown premise, and avoid adding identity/proof conditions absent from the task contract. These remain review recommendations; no skill-wide activation or new DPO/SFT admission follows from them.

## Corpus inventory

Current published admitted facets:5773 native (3738train/2035test),4962 text (3258/1704),5004R (3115/1889). Counts are overlapping facets, not independent task/world counts. The live GPU run used the immutable4880 text input; publication does not mutate it. Fresh sampled actions remain held until explicit per-action review and composition adoption.

- Full4793 historical text documents were rebuilt from exact serving request/assistant boundaries;87 newly adopted documents complete4880. Split/group and tokenizer checks are preserved.
- Registered target-bound refresh and convenience input are verified on Pop and DGX. Convenience corpus: `luna-v17-foundation-text-input-4880-20261009-v1`; inputSHA `b10da15de2510b55635b5aaa85301204e559669ac239968bc94e4393d08e0572`.
- Shared sync copies verified file aliases as bytes; immutable selected-file manifests must exclude private process databases/logs/caches. Unsafe retired manifests remain untracked and held. Do not commit or reactivate them.
- Never silently switch the active training input. Publish fresh facets through registry/admission/provenance first, then explicitly declare the next recipe or continuation.
