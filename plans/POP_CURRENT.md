# Pop current work

Last reviewed: 2026-10-09 11:06 UTC. This page is a navigation aid; inspect live processes and receipts before acting. Historical evidence and course changes remain in [HANDOVER.md](HANDOVER.md) and [GENERATION_DECISIONS.md](GENERATION_DECISIONS.md).

## Ownership and coordination

- Pop owns execution here; DGX owns its execution. Canonical checkout on both: `/home/werg/natlang`. Do not launch from retired mirrors or change DGX jobs.
- Run `python3 scripts/coordination_inbox.py check --ack` at session start, before resource changes and each monitoring cycle. Use Git for code and the corpus registry/manifests plus `sync_training_corpora.py` for selected data artifacts.
- The user wants autonomous work and monitoring, including waiting/sleeping when jobs are in progress. Keep five authorized Luna slots supplied where appropriate. Do not report planned workers as actual provider activity.

## GPU: active foundation continuation

Container: `neuralese-foundation-text-4880-continuation-20261009-v1`.

Recipe: [luna-foundation-text-4880-continuation-from-repair-best-20261009-v1.json](../training/neuralese/recipes/luna-foundation-text-4880-continuation-from-repair-best-20261009-v1.json).

Output: `/srv/storage/natlang-artifacts/neuralese-foundation-text-4880-continuation-20261009-v1/direct-stage-run/foundation_text_continuation`.

- Full-state continuation from repair best1280: model, distinct full/sketch projections, Muon/AdamW optimizer, schedule and RNG restored. It uses all 4880 admitted target-bound text documents, not the old 87-document repair subset. Absolute step cap4096; no automatic qualification from completing steps.
- Latest observed evaluation/current checkpoint1792; best receipt1664. Inspect `best-checkpoint.json`, not training log progress, to choose saved weights. GPU approximately100%,4939MiB of8188MiB.
- Eval1792 full/shallow raw-embedding relativeMSE .1262/.5858; repeated-pass excessCE .820/1.251. Alignment gate **false**. Both projections train against raw next-token embeddings; backbone adaptation is active; three sequence passes use shifted projected outputs.
- Full-depth projected autoregression matched crisp on one256-token control at1664 but diverged at index3 at1792; crisp remained exact. The narrow success is unstable. Sketch autoregression remains poor. Do not treat this as qualification or move to recurrence merely because text warm-up reached its cap.
- Next: monitor held metrics and storage; diagnose exact-best foundation functionality if improvement stalls. Qualify runtime transport/gradient replay separately against the actual selected weights before recurrence.

### Shutdown and storage

The active immutable runtime predates the repaired recipe signal forwarding. If an early stop is necessary, signal its ancestry-verified trainer child and verify the emergency checkpoint; do not rely on Docker stopping the outer PID1. Normal completion is safe. Future launches should use shared signal fix18770099; actual CPU-only Docker routing proof is in `runs/neuralese-recipe-signal-forwarding-proof-20261009-v1/receipt.json`.

Old repair Docker shutdown lost73 unsaved updates: serialized current1920 versus final log1993. Preserve current1920, chosen best1280, incident receipt and original recurrence best1024. No emergency checkpoint was produced by that failed stop.

Available space approximately root14GiB / external5.7GiB, including a1.436GB checkpoint reserve. Current/best are different files after1792; avoid redundant checkpoint/model copies. Historical Step5 process state and unused grouped Ling base weights were handled with verified local offload/duplicate eviction; private process/browser caches are not training corpora.

## Generation and review

### Five Luna workers

Root: `runs/luna-v6-criterion-fivecase-diagnostic-20261009-v1`.

- All five queue runners and provider activity confirmed. Three Bridge/River/Emergency train sources explicitly use bound v6 criterion/evidence skills; two original Archive/Wetland train controls.
- Exact source/skill proofSHA `5939dc1fabc50187a85577e3f8ee8aa6e548d2dc44ac8ba4a90e2fff1b75e25b`; planSHA `aaf075d1cd9039fb23b8873f2aa419c6576a9e1db352ef83913722821bcccdb6`.
- Active plan's statement that root code is unchanged is corrected separately in `campaign-live-v1/source-provenance-correction-v1.json`: diagnostic prompt code changed; facts/oracles/groups/splits did not. This tests combined skill and explicit-use treatment, not skill-only causal effect. Controls have pinned original oracle references; their fresh generic fake fixture was incompatible with a Neuralese marker and did not pass runtime replay.
- Review every observed action and skill disclosure before admission. No global v6 activation, new-world credit or blanket task-level admission.

### Step5 Preview Free

`runs/step5-preview-free-clinic-row0-corrected-20261009-v1` completed the correct Clinic source0/scenario2 and passed its oracle. It has18 actions,15 children and6 iterations. Owned idle bridge stopped; captures sealed. One malformed `return_result({code:...})` attempt is held separately; subsequent valid typed output is independently reviewed. Existing modern adapters are preparing native/text/recurrence candidates; root replay/adoption required before publication. Keep Step5 on a fresh distinct train source after exact-source launch checks. No paid fallback or text-distillation flag.

### Failure repairs and skills

- Exact-context source audit confirms three genuine positive-eligibility false judgments (BIR-2C, RWA-1C, ESR-2A), with evidence visible and faithful host capture. These are not converter/read/transport failures.
- Shared `build-preference-pairs.mjs --source-derived-repairs PROPOSAL.json` now preserves three checked counterfactual repairs in a held sidecar. Ordinary preference output is empty. Synthetic chosen targets are explicit; no task/hidden-state equivalence or DPO admission. Per-item review/promotion remains unfinished; see latest handover pins.
- V6 `judge-against-criteria` adds a generic positive conjunction example; other six v5 skills unchanged. Tracked diagnostic authorization scopes its current use. Evaluate results and discovery before broader activation/SFT.

## Corpus inventory

Current admitted facets:5684 native (3649train/2035test),4880 text (3176/1704),5004R (3115/1889). Counts are overlapping facets, not independent task/world counts. Fresh generation/review packets are excluded until explicit adoption.

- Full4793 historical text documents were rebuilt from exact serving request/assistant boundaries;87 newly adopted documents complete4880. Split/group and tokenizer checks are preserved.
- Registered target-bound refresh and convenience input are verified on Pop and DGX. Convenience corpus: `luna-v17-foundation-text-input-4880-20261009-v1`; inputSHA `b10da15de2510b55635b5aaa85301204e559669ac239968bc94e4393d08e0572`.
- Shared sync copies verified file aliases as bytes; immutable selected-file manifests must exclude private process databases/logs/caches. Unsafe retired manifests remain untracked and held. Do not commit or reactivate them.
- Never silently switch the active training input. Publish fresh facets through registry/admission/provenance first, then explicitly declare the next recipe or continuation.
