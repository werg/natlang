# Pop current work

Last reviewed: 2026-10-09 11:56 UTC. This page is a navigation aid; inspect live processes and receipts before acting. Historical evidence and course changes remain in [HANDOVER.md](HANDOVER.md) and [GENERATION_DECISIONS.md](GENERATION_DECISIONS.md).

## Ownership and coordination

- Pop owns execution here; DGX owns its execution. Canonical checkout on both: `/home/werg/natlang`. Do not launch from retired mirrors or change DGX jobs.
- Run `python3 scripts/coordination_inbox.py check --ack` at session start, before resource changes and each monitoring cycle. Use Git for code and the corpus registry/manifests plus `sync_training_corpora.py` for selected data artifacts.
- The user wants autonomous work and monitoring, including waiting/sleeping when jobs are in progress. Keep five authorized Luna slots supplied where appropriate. Do not report planned workers as actual provider activity.

## GPU: active foundation continuation

Container: `neuralese-foundation-text-4880-continuation-20261009-v1`.

Recipe: [luna-foundation-text-4880-continuation-from-repair-best-20261009-v1.json](../training/neuralese/recipes/luna-foundation-text-4880-continuation-from-repair-best-20261009-v1.json).

Output: `/srv/storage/natlang-artifacts/neuralese-foundation-text-4880-continuation-20261009-v1/direct-stage-run/foundation_text_continuation`.

- Full-state continuation from repair best1280: model, distinct full/sketch projections, Muon/AdamW optimizer, schedule and RNG restored. It uses all 4880 admitted target-bound text documents, not the old 87-document repair subset. Absolute step cap4096; no automatic qualification from completing steps.
- Latest observed evaluation/current/best checkpoint2560. GPU approximately100%,4943MiB of8188MiB. Inspect saved checkpoint receipts before selecting weights.
- Held full/shallow raw-embedding relativeMSE .1210/.5396; long-tail repeated-pass excessCE .618/.964 and agreement .870/.810. Alignment gate **false**; improving held metrics do not yet qualify recurrence.
- One256-token full-depth projected autoregressive control is exact at2560, but earlier saved checkpoints alternated between exact and divergence at index3. This is a fixed control, not randomized window selection. Sketch autoregression remains poor. New shared diagnostics fingerprint control windows and distinguish preceding feedback from current emitted payload; the active frozen runtime predates those additions.
- Next: monitor held metrics and storage; diagnose exact-best foundation functionality if improvement stalls. Qualify runtime transport/gradient replay separately against the actual selected weights before recurrence.

### Shutdown and storage

The active immutable runtime predates the repaired recipe signal forwarding. If an early stop is necessary, signal its ancestry-verified trainer child and verify the emergency checkpoint; do not rely on Docker stopping the outer PID1. Normal completion is safe. Future launches should use shared signal fix18770099; actual CPU-only Docker routing proof is in `runs/neuralese-recipe-signal-forwarding-proof-20261009-v1/receipt.json`.

Old repair Docker shutdown lost73 unsaved updates: serialized current1920 versus final log1993. Preserve current1920, chosen best1280, incident receipt and original recurrence best1024. No emergency checkpoint was produced by that failed stop.

Available space approximately root14GiB / external5.7GiB, including a1.436GB checkpoint reserve. Current/best are different files after1792; avoid redundant checkpoint/model copies. Historical Step5 process state and unused grouped Ling base weights were handled with verified local offload/duplicate eviction; private process/browser caches are not training corpora.

## Generation and review

### Five Luna workers

Root: `runs/luna-v6-criterion-fivecase-diagnostic-20261009-v1`.

- Three five-case waves have completed; latest wave3 reseeds all five source-bound train cases passed their runtime outcomes. Results remain held pending per-action review; accepted case output is not training admission.
- Latest wave3 plan: `extension-wave-v1/dispatch-wave3-reseed-v1/dispatch-plan-v1.json`, SHA49e99f4b1b35e68561e9c4e6a38620662ac23500ff485829f8f92a44abc370ff. It pins exact overlay source4bc9…, current runtime7532… and aggregate proofbcd7… separately from original source visibility proof5939…. Future plans should name the aggregate as their primary source proof.
- Next five tasks come from V26 editing, archive, grant and records train sources. Its previous apparent “generation” was static reference proof, not model trajectories. Current-runtime preflight exposed a fixture matching filenames inside carried historical text instead of the current FileHandle preamble. Selector-only fixture correction passes16/16; runtime access boundaries, gold and source splits are unchanged. Preserve old failures and bind fresh exact proof before launching.
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
