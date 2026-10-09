# Pop current work

Last reviewed: 2026-10-09 14:24 UTC. This page is a navigation aid; inspect live processes and receipts before acting. Historical evidence and course changes remain in [HANDOVER.md](HANDOVER.md) and [GENERATION_DECISIONS.md](GENERATION_DECISIONS.md).

## Ownership and coordination

- Pop owns execution here; DGX owns its execution. Canonical checkout on both: `/home/werg/natlang`. Do not launch from retired mirrors or change DGX jobs.
- Run `python3 scripts/coordination_inbox.py check --ack` at session start, before resource changes and each monitoring cycle. Use Git for code and the corpus registry/manifests plus `sync_training_corpora.py` for selected data artifacts.
- The user wants autonomous work and monitoring, including waiting/sleeping when jobs are in progress. Keep five authorized Luna slots supplied where appropriate. Do not report planned workers as actual provider activity.
- Current shared-work ownership: DGX is leading the shared objective/schedule and recipe-launcher consolidation; Pop owns the C++/WASM read-port transport for the reader adapter. Coordinate before editing shared renderer/window helpers. The current Pop map run is frozen and does not include later shared-runtime changes.

## GPU: mapped-input foundation continuation

The superseded sketch-consumer repair is closed and is artifact evidence only;
its best step3328 is not a qualified source. The active Pop job is the mapped-input continuation
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
heldout, SHA `b10da15de2510b55635b5aaa85301204e559669ac239bc94e4393d08e0572`;
the admitted87 target-bound documents are already in it and must not be appended
again. Native records/pieces are source bindings, not extra text rows.

At the latest verified monitoring (14:24 UTC), the serialized current checkpoint
is step23680, the best checkpoint remains step23040, and training had reached
step23754 in the log.
The latest gate is false with zero consecutive passes; this continuation has not
inherited qualification from its source checkpoint. At step23680, held input-map
and full-depth embedding errors were 0.11274 and 0.11679, with pass-1 CE delta
0.000584. The map/full error gap has narrowed from 0.02555 at step22272 to
0.00405, but strict alignment still fails. At step23680, pass-1 token agreement
was 0.97802 for assistant reasoning, 0.98779 for tool content, 0.99688 for user
content, and 0.99759 for assistant replies. These measurements support moving
toward real autoregressive exposure after the warm-in rather than waiting for
exact map/full parity; they do not qualify autoregressive behavior, transport,
stopping, or task execution. GPU was 100% at 4729 MiB; free space was 8.5 GiB
on `/` and 5.1 GiB on `/srv/storage`. Driver output is under
`/srv/storage/natlang-artifacts/neuralese-map-refresh-4880-from-qualified-22272-20261009-v5/preflight-v1/`.
The recipe launch preflight binds code, input, image, cache/token mounts and
resource floor. Continue monitoring exact current/best checkpoint metadata and
fresh evaluation results; do not reuse results from the old data or sketch run.

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

Latest storage check: 8.5 GiB free on `/` and 5.1 GiB free on `/srv/storage`; the active run keeps its checkpoint reserve. Its current and best full checkpoints are distinct. Avoid redundant checkpoint/model copies. Historical Step5 process state and unused grouped Ling base weights were handled with verified local offload/duplicate eviction; private process/browser caches are not training corpora.

## Generation and review

### Five Luna workers

Root: `runs/luna-v6-criterion-fivecase-diagnostic-20261009-v1`.

- The recent held packet has been reduced from199 proposals to22 after two definite visibility holds. All22 remaining proposals are held and unadmitted; runtime outcomes or packet-level acceptance do not grant training admission.
- Earlier wave3 plan and its source/runtime proofs remain preserved under this campaign. Treat them as historical evidence; do not use their previous “latest wave” wording as current status.
- Review every observed action and skill disclosure before admission. No global v6 activation, new-world credit or blanket task-level admission.

### Step5 Preview Free

`runs/step5-preview-free-clinic-row0-corrected-20261009-v1` completed the correct Clinic source0/scenario2 and passed its oracle. It has18 actions,15 children and6 iterations. Owned idle bridge stopped; captures sealed. One malformed `return_result({code:...})` attempt is held separately; subsequent valid typed output is independently reviewed. Current shared adapters produced17 reviewed native actions and17 exact-serving-boundary text documents; root native and text replays are byte-identical. One action is an authored static root and16 are model samples; actor provenance is now mandatory in shared conversion. Token-suffix audit passed and root composition adopted17native/17text; curated publication/sync remain pending; no recurrence or hidden-state admission. Step5 Microgrid terminated on repeated endpoint-unavailable503 errors, with captures sealed under `runs/step5-microgrid-concurrent-luna-wave2-20261009-v1`. No paid fallback or text-distillation flag.

### Failure repairs and skills

- Exact-context source audit confirms three genuine positive-eligibility false judgments (BIR-2C, RWA-1C, ESR-2A), with the relevant facts visible and faithful host captures. The failure boundaries were respectively a false leaf answer, an added identity/proof requirement in the delegated criterion, and a stale “eligibility unknown” premise passed to a child. The original three repair rows remain held.
- The shared `build-preference-pairs.mjs --source-derived-repairs PROPOSAL.json` adapter now accepts the exact-context v5 causal-boundary proposal. It joins source row, request hash, raw response hash, terminal tool-call ID, trace action and offered schema; it checks chosen eval syntax and names against declarations in the captured opening scope only, not the full persistent eval history. All three candidate pairs stay in the held sidecar pending per-item root preference admission; ordinary preference output remains empty. The chosen side preserves the eligibility delegation, and carries no task-success, runtime-equivalence or hidden-state claim.
- The audit's generic skill recommendations are to evaluate each stated conjunction against supplied facts, use later source-bound evidence instead of retaining a superseded unknown premise, and avoid adding identity/proof conditions absent from the task contract. These remain review recommendations; no skill-wide activation or new DPO/SFT admission follows from them.

## Corpus inventory

Current admitted facets:5701 native (3666train/2035test),4897 text (3193/1704),5004R (3115/1889). Counts are overlapping facets, not independent task/world counts. Fresh generation/review packets are excluded until explicit adoption.

- Full4793 historical text documents were rebuilt from exact serving request/assistant boundaries;87 newly adopted documents complete4880. Split/group and tokenizer checks are preserved.
- Registered target-bound refresh and convenience input are verified on Pop and DGX. Convenience corpus: `luna-v17-foundation-text-input-4880-20261009-v1`; inputSHA `b10da15de2510b55635b5aaa85301204e559669ac239968bc94e4393d08e0572`.
- Shared sync copies verified file aliases as bytes; immutable selected-file manifests must exclude private process databases/logs/caches. Unsafe retired manifests remain untracked and held. Do not commit or reactivate them.
- Never silently switch the active training input. Publish fresh facets through registry/admission/provenance first, then explicitly declare the next recipe or continuation.
