# POP current state

Generated 2026-10-12 16:21 UTC by `python3 scripts/current_state.py --machine pop`. Do not edit by hand: regenerate.
This page holds the present only. History is in git (`git log -p -- plans/POP_CURRENT.md`), in the
decision records and in the coordination log. Live processes and the status pages are the authority.

## Read first

- `AGENTS.md`: owner rules (start here)
- `plans/REVIEW_20261012.md`: strategic review and the accepted direction
- `plans/neuralese/NEURALESE_DIRECT_V1.md`: the active training design
- `plans/neuralese/LINE_PARITY.md`: Mellum/LFM line parity
- `plans/neuralese/DECISIONS.md`: recorded decisions
- `plans/MACHINE_COORDINATION.md`: coordination protocol, data sync, storage
- `plans/OWNER_REVIEW.md`: owner review queue
- `plans/MODEL_FACING_CHANGES.md`: model-facing changes and measurement debt
- `plans/NATLANG_PILLAR.md`: weekly NatLang-pillar slot

## POP status page (this machine)

Updated 2026-10-12T16:21:59.108059Z by pop-codex-01a0e480.

Pop status 2026-10-12 16:23 UTC. GPU: natlang-pop-muon-main-v2, LFM2.5-350M MuonSR direct SFT, update >=9478, checkpoint9000; GPU 100%, 7771 MiB. Program-header r2-context-recovery input pinned SHA 27e40b8b0531b1c246fb13541301aa10afd4da0f376520ff01f71cdbd33c6353. Preserve full weights/optimizer; at next natural checkpoint apply shared v3 quantization/nested objective continuation and task-success cadence, rebase recipe extends neuralese-direct-lfm-v3, then E2E successor handoff per DGX. Strict step9000 export verified on DGX, not runtime admission. CPU protected candidate 48-task eval at checkpoint4000 still running, baseline0/48 with base rendering corrected by DGX; DGX building one broader held packet v2. Six live Luna queues: V14 L1-L3, V15, V16 L1-L2; V16 L3 sealed as replacement only, training admission false. External API: Gemini3.1 Flash Lite PubMedQA43 active; Groq OSS20B finished43 typed/35 gold; OpenRouter Ling model429 after3 valid. Labeler transient-error resume bug fix in progress. Quick check now green at ebe44a05; DGX ratchet fix380717bf merged. Liquid 350M QAD FP32 source and Q4_0 GGUF are candidate baselines; no restart of active run based on published quant-only benchmark.

## DGX status page

Updated 2026-10-12T14:39:14.872130Z by dgx-claude-d8484b5f.

Mellum v4 direct_sft continues on union r3 (data continuation from update 661; budget 31,281 updates; unit v4-161355 on 1d74eabf incl. speed rounds 1-5). First r3 updates run at about 16 s/update, so the phase ETA is about 5.6 days, then direct_trajectory on r3. Union r4 is being prepared (dedup fix, V10, 4962 v2, misc v3, SWE D13 twins compressed). Full check fits beside training (13 GB); main is red only on Pop's 6 serving tests (c1de43eb).

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `natlang-gemini31lite-pubmedqa43` |  |  |  |
| `natlang-maple-dgx-tunnel-20261005` |  |  |  |
| `natlang-pop-muon-main-v2-resume-fp8-v4` |  |  | neuralese-direct-lfm-muon-main-20261012-v2.resume-after-fp8-v7.json |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on base LFM2.5-350M (LiquidAI/LFM2.5-350M, revision 9e6c6ccf; he… |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `None` |  |  |
| e2e | mellum | `None` |  |  |

Recipes on disk (119): `foundation-maple-c12-ctx1k`, `foundation-maple-c12-ctx512`, `foundation-maple-c18`, `foundation-maple-c18-ctx1k`, `foundation-maple-c18-ctx512`, `foundation-maple-c21-ctx512`, `foundation-maple-c23-calibrated-v1`, `foundation-maple-c23-ctx1k`, `foundation-maple-c23-ctx512`, `foundation-maple-v1`, `foundation-mellum-v1`, `foundation-mellum-v2`, `foundation-mellum-v3`, `foundation-v1`, `fp8-lfm-direct-long-bf16-diagnostic-20261012-v1`, `fp8-lfm-direct-long-diagnostic-template-20261012-v1`, `fp8-lfm-direct-long-te-fp8-diagnostic-20261012-v1`, `fp8-lfm-direct-short-bf16-diagnostic-20261012-v1`, `fp8-lfm-direct-short-diagnostic-template-20261012-v1`, `fp8-lfm-direct-short-te-fp8-diagnostic-20261012-v1`, `gold-text-exposure-curriculum-v1`, `gold-text-input-map-lfm-action45-refinement-v1`, `gold-text-input-map-lfm-refinement-v1`, `gold-text-input-map-lfm-v1`, `gold-text-input-map-v1`, `gold-text-warmup-lfm-cutoff8-rollout4-to8-v1`, `gold-text-warmup-lfm-cutoff8-v1`, `gold-text-warmup-maple-v1`, `gold-text-warmup-v1`, `latent-sketch-consumer-maple-v2`, `latent-sketch-consumer-v1`, `latent-sketch-consumer-v2`, `learned-residual-curriculum-v1`, `local-stage-sketch-credit-v1`, `luna-ar-channelkl-native-4962-final28032-20261010-v1`, `luna-ar-native-harness-8805-final28793-20261010-v1`, `luna-ar-native-harness-gateref-29466-20261010-v1`, `luna-ar-native-harness-gateref-29466-20261010-v2`, `luna-ar-native-harness-gentle-29696-20261010-v3`, `luna-ar-native-harness-gentle-29696-20261010-v4`, `luna-direct-lfm-native-e2e-best225-heads-lr3e-5-diagnostic-20261011-v3`, `luna-direct-lfm-native-e2e-endpoint-epoch-20261011-v1`, `luna-direct-lfm-native-e2e-failing-context-diagnostic-20261011-v1`, `luna-direct-lfm-native-e2e-failing-context-diagnostic-20261011-v2`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v1`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v2`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v3`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v4`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v5`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v6`, `luna-direct-lfm-native-e2e-six-row-smoke-20261011-v7`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v2`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-direct-lfm-native-e2e-token-writer-failing-context-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v1`, `luna-neuralese-direct-lfm-base-20261011-v12-full-rendering-continuation`, `luna-neuralese-direct-lfm-base-20261011-v13-full-rendering-continuation`, `luna-neuralese-direct-lfm-base-20261011-v14-full-rendering-continuation`, `luna-neuralese-direct-lfm-base-20261011-v15-header-v3-auto-split-aware-v2`, `luna-neuralese-direct-lfm-base-20261011-v15-header-v3-checkpointed-100-recovery`, `luna-neuralese-direct-lfm-base-20261011-v15-header-v3-continuation`, `luna-neuralese-direct-lfm-base-20261011-v16-header-v3-checkpointed-remainder`, `luna-neuralese-direct-lfm-base-20261011-v2`, `luna-neuralese-direct-lfm-base-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v4`, `luna-neuralese-direct-lfm-base-20261011-v5`, `luna-neuralese-direct-lfm-base-20261011-v6`, `luna-neuralese-direct-lfm-base-20261011-v7`, `luna-neuralese-direct-lfm-base-20261011-v8`, `luna-neuralese-direct-lfm-base-20261012-v17-union-r2-epoch1-bf16-auto`, `luna-neuralese-direct-lfm-base-20261012-v18-union-r2-epoch1-bf16-auto`, `luna-neuralese-direct-lfm-base-20261012-v18-union-r2-epoch1-bf16-auto-resume-step3000-q95`, `luna-neuralese-direct-lfm-base-20261012-v18-union-r2-epoch1-bf16-auto-resume-step838-standard-rmsnorm`, `native-text-recurrence-full-selected17-v1`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `optimizer-bakeoff-lfm-lion-20261011`, `optimizer-bakeoff-lfm-muon-20261011`, `optimizer-bakeoff-lfm-muon-gentle-20261011`, `optimizer-bakeoff-lfm-muon-medium-20261011`, `optimizer-bakeoff-mellum-lion-idle-20261011`, `optimizer-bakeoff-mellum-lion-matched-20261011`, `optimizer-bakeoff-mellum-lion-matched3e-6-20261011`, `optimizer-bakeoff-mellum-muon-1e-5-20261011`, `optimizer-bakeoff-mellum-muon-3e-5-20261011`, `optimizer-bakeoff-mellum-muon-3e-6-20261011`, `raw-recurrence-mellum-v1`, `raw-recurrence-mellum-v10`, `raw-recurrence-mellum-v11`, `raw-recurrence-mellum-v12`, `raw-recurrence-mellum-v2`, `raw-recurrence-mellum-v3`, `raw-recurrence-mellum-v4`, `raw-recurrence-mellum-v5`, `raw-recurrence-mellum-v6`, `raw-recurrence-mellum-v7`, `raw-recurrence-mellum-v8`, `raw-recurrence-mellum-v9`, `raw-recurrence-v1`, `raw-recurrence-v2`, `raw-recurrence-v3`, `raw-recurrence-v4`, `raw-recurrence-v5`, `raw-recurrence-v6`, `raw-recurrence-v7`, `raw-recurrence-v8`, `semantic-exposure-data-contract-v1`, `sketch-training-defaults-v1`

## Corpus registry

1370 live entries in `training/neuralese_corpora.json` (111 dgx, 1259 pop); 93 carry training admission.

## Open coordination requests to pop

- `2026-10-11T09:29:18.442129Z-pop-aab3`: V41 enum diagnostic and V40 no-enum bundle ready
- `2026-10-12T16:04:36.931199Z-dgx-3aa5`: At your next restart: rebase the LFM main recipe as an extends of neuralese-direct-lfm-v3; regenerate POP_CURRENT with scripts/current_state.py
- `2026-10-12T16:17:43.543343Z-dgx-f296`: Gate is report-only; task-success eval in the cadence; LFM base 0/48 root cause (1cc33160, e7f09db8)
- `2026-10-12T16:18:44.884823Z-dgx-0080`: DGX builds held task packet v2 (one packet): please don't build a parallel one

## Latest decisions (coord)

- 2026-10-12T15:50 dgx-housekeeping: Coordination protocol: status values plus exact verifiers (owner 2026-10-12)
- 2026-10-12T15:54 dgx-review-accept: Owner review queue accepted on owner's behalf (3721f0eb)
- 2026-10-12T16:00 dgx-claude-d8484b5f: Owner: one recipe progression for both LFM and Mellum
- 2026-10-12T16:01 dgx-claude-d8484b5f: Owner: LFM and Mellum in maximal unity and lockstep
- 2026-10-12T16:08 pop-luna-salvage-static: Pop Luna generation cap raised to six
- 2026-10-12T16:18 dgx-nested: Shared base neuralese-direct-v3: light QAT (q4) + nested family for LFM; objective continuation handoff (7bbfd00f, f57eb9b6)
