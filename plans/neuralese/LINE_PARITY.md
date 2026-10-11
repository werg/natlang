# Line parity: Mellum (DGX) and LFM (Pop)

Owner, 2026-10-11: both lines use the same recipe, data and code. Settings may differ only to fit a machine, or
where the backbone requires it, and each such difference is declared with its reason. "Let's rethink things and make
sure we're not accidentally drifting apart. And let's unify things towards the better."

How it is kept:
- **Shared bases.** `recipes/neuralese-direct-v2.json` is the direct stage's shared base (data, context, monitor,
  cadence). The end-to-end stage's shared base is `stages.e2e.base` in `training/neuralese/line-parity.json`
  (data and parameters).
- **Declared differences.** `training/neuralese/line-parity.json` lists each line's next-launch recipe and every
  allowed difference (`key`, `kind` fit-only | backbone-inherent, `reason`).
- **Exact check.** `natlang_neuralese.train.line_parity` (in `scripts/check-main.sh --quick`, via
  `tests/neuralese/test_line_parity.py`) resolves each line's effective configuration and fails on any undeclared
  difference, on a declared difference that no longer differs, and on unequal supervision
  (`steps x target_tokens x batch`).
- **Launch refusals.**
  - The recipe CLI refuses `--records/--pieces/--text-data` for a recipe that declares its data; files move only
    with `--input-binding NAME=PATH`, which keeps the sha256.
  - The direct-stage runner refuses a fresh `neuralese_direct` or direct-artifact recurrence launch unless the
    recipe declares `"parity": {"line": "mellum"|"lfm", "stage": "direct"|"e2e"}` (checked against the base) or
    `"parity": "diagnostic"`.
- **Code.** Both lines launch from current `origin/main`. Pop pins the frozen package of each launch; DGX runs a code
  snapshot of a main commit. A run on older code is recorded as such; it does not change the shared base.

## Audit, 2026-10-11

Compared: Mellum `neuralese-direct-mellum-v2` as run by v2c (`runs/neuralese-direct-mellum-20261011/run-v2`) against
Pop's `luna-neuralese-direct-lfm-base-20261011-v8` (direct), and Pop's
`luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v2` against DGX's Mellum E2E smoke launcher
(`runs/neuralese-direct-mellum-20261011/e2e/launch-e2e-smoke.sh`).

Classes:
- **fit-only:** memory, hardware or time;
- **backbone-inherent:** required by the backbone;
- **accidental:** neither. The better side moves into the shared base.

### Direct stage

| difference | Mellum v2c | LFM V8 | class | chosen (shared base unless noted) | reason |
|---|---|---|---|---|---|
| data | V9 records rendered by `gold_text_rows`: 4480 docs (2776 train / 1704 held) | 4962-doc root-adopted text union (3258 / 1704) | accidental | the 4962 union, declared in the base; Mellum binds its tokenizer twin | The union is the reviewed superset: all 4480 V9 documents plus 482 later reviewed train documents, with identical held documents (checked by source-record keys, `runs/neuralese-direct-line-parity-20261011/count-v9-mellum.json`). |
| data rendering | Mellum tokenizer/template | LFM tokenizer/template | backbone-inherent | registered tokenizer twin (same ids, splits, source groups, source records) | Prepared text carries token ids. |
| tokens (context window) | 8192 (context ≤ 7168) | 32768 (≤ 32512) | accidental | 32768 | Every union document keeps its whole preceding context (Mellum max 13,366 tokens; Mellum2.1 supports 131,072 positions via YaRN). At 7168, 3.1% (Mellum) / 4.3% (LFM) of supervised tokens lost early context. |
| target_tokens | 1024 | 256 | fit-only | Mellum 1024, LFM 256 | Pop's 8 GB GPU; each token is supervised once either way. |
| steps | 2000 | 1000 | accidental (8x less supervision on LFM) | equal supervised positions: 750 x 1024 = 768,000 (about 2 epochs); LFM 3000 steps (fit-only consequence) | Revised 2026-10-11 (coordinator): about 2 epochs of the ~384k supervised tokens (more erodes instruction following, arXiv:2609.01244); v2 plateaued near update 1800 with Lion and MuonSR learns several times faster per update. Mellum cadence: eval at 375 and 750, drift every 150, gate-at-warn at most every 450; LFM 3000/1500/750/2250. |
| latent optimizer | LionSR | LionSR | shared | MuonSR (cubic5 Newton-Schulz, Nesterov 0.95, 0.2·sqrt(max) scale, wd 0.1, idle experts skipped) for both | Owner tie-break 2026-10-11: a cheap, battle-tested Muon for both lines when comparable; Mellum2 was pretrained and SFT'd with Muon. One implementation, not a per-backbone split (OPTIMIZER_BAKEOFF_20261011.md). |
| latent_lr (MuonSR) | 1e-5 | 3e-6 (shared base value) | backbone-inherent (evidence) | per line | LFM MuonSR 3e-6: gate 26/48 = base at 300, held CE 6.19 to 1.05 (1e-5 drift-stopped under the old 0.5 bound). Mellum MuonSR 1e-5: gate 0.75 = base at 100, response KL 0.30, train CE per hour at or above LionSR 3e-6. |
| drift policy | gate + KL stop 0.5 | gate + KL stop 0.5 | shared | behaviour gate (≥ base − 0.1) is the hard criterion; response KL warns at 0.5 (runs the gate), stops at 1.5 | Coordinator decision 2026-10-11: SFT is meant to change the distribution; the gate measures forgetting. |
| backbone_lr / latent_lr (LionSR era) | 7.5e-6 / 3e-6 | 7.5e-7 / 3e-7 | backbone-inherent (evidence) | backbone_lr per line (latent_lr: rows above) | LFM at the shared rates drift-stopped at update 260 (V6: gate 16/48, base 26/48). At 1/10 it finished (V8: gate 22/48, response KL 0.018). Mellum at the shared rates qualified (run-v1) and v2 holds gate 0.79 vs base 0.75 at update 1300. |
| drift_every, drift_gate_every, drift_gate_rise, drift_ce_windows, gate_response_kl | 100, 300, 0.05, 4, 0.5 | absent (V8's code predates them) | accidental (code age) | current shared code and values; cadence per the policy below | They are fixes that came after V8's frozen package (bdb6851b). |
| eval_every / checkpoint_every | 2000 / 1000 (+ drift every 100) | 20 / 100 | accidental | cadence policy below | Owner: no super-frequent evals or saves (25% wall overhead on Pop). |
| drift reference | quantization `preserve.teacher` held rows (Mellum teacher v3) | in-run capture on `monitor.data` (48 probes) | accidental | in-run capture on the shared 48 probes | One mechanism that works on any backbone, so the reference is the actual base model's answers. |
| behaviour gate probes | 24 (6 each plain, thinking, tool, natlang), quantization.gate | 48 (12 each plain exact math, tool, open answer, natlang test turn), monitor.gate | accidental | the 48 probes for both | Twice the probes (less noise), exact final-answer checks, real native test turns. Mellum's 6 thinking probes leave the shared gate (a known coverage loss; a shared thinking probe set can be added for both lines). |
| gate reference gate_pass | 0.75 (18/24) | 0.5417 (26/48) | backbone-inherent | each line measures its base on the 48 probes | It is a measurement of the base model. Mellum base: 28/48 = 0.583 (plain math 10/12, tool 12/12, natlang test turns 5/12, open answers 1/12, which run past the 256-token budget). |
| gate output convention | (24-probe set: maple) | lfm (declared per probe) | backbone-inherent | gate-level `output_convention` (new shared key); Mellum `maple` | Under the probes' LFM parser, Mellum's well-formed `<tool_call>{json}` calls scored 0/12. That was a checker bug, not behaviour. |
| gate KL bound | 0.25, response (`held_kl_bound`), overridden by `gate_response_kl` 0.5 | 0.5, combined | accidental (code age) | shared `gate_response_kl` 0.5 on response positions | The accepted direct-v1 rule is response-only (58e901f6, b1cd6632). |
| moe_kernel | grouped | n/a | backbone-inherent | Mellum only | Dense MoE experts as grouped GEMMs. |
| quantization component | raw-recurrence-mellum-v11 points | none | backbone-inherent + declared gap | Mellum keeps it; LFM's is a known gap | The owner's QAT rule applies to LFM once a component is declared. |
| heads | raw-token-v1 heads from base (`foundation_heads`) | untrained clean-base heads | lineage | each line's own | Both are raw-token heads from the base model. |
| code | snapshot bc666539 (17:28) | frozen package of bdb6851b (12:52): 18 modules older, incl. before a8972488 (attention latents), 7c6eea80 (drift_every), 58e901f6 (response-only gate KL) | accidental (age) | current main at launch | Newest code always (owner). |
| held_documents, prefix_tokens, batch, weights, projection shape, optimizer kinds, drift warn/stop, value_margin, seed, gate_held_rows | equal | equal | — | — | — |

### End-to-end stage

| difference | Mellum (smoke launcher) | LFM endpoint epoch v2 | class | chosen | reason |
|---|---|---|---|---|---|
| records | V9 assembled-v3 native records, `--only-handover`, train 400 / eval 16 | 1243-row authenticated binding cohort (687 train / 32 held) | accidental | the 1243 cohort | It is the reviewed, authenticated binding set; the V9 set predates it. |
| pieces | V9 native pieces | cumulative-full-assembly-v3 recurrence pieces | accidental | cumulative-full-assembly-v3 recurrence pieces | They match the cohort. |
| max_tokens | 8192 | 32768 | accidental | 32768 | Same reason as the direct stage. |
| max_write_vectors, writer_text_weight, context/feedback weights, seed | absent/defaults | 2048, 1.0, 1.0/0.25, 20261011 | accidental | LFM's values | They were declared by the only run with training evidence. |
| lr (soft parameters) | 1e-4 | 3e-4 | accidental | 3e-4 | Same evidence. |
| backbone_lr / qat_latent_lr | (7.5e-6) / 3e-6 | 7.5e-7 / 3e-7 | backbone-inherent | as in the direct stage | Same evidence as the direct stage. |
| latent optimizer / rate (2026-10-11) | LionSR (`qat_latent_lr`) | LionSR 3e-7 | shared | MuonSR, the direct stage's implementation and options (`latent_optimizer muonsr`, `latent_lr` per line: Mellum 1e-5, LFM 3e-6 = base; cubic5, wd 0.1) | One latent optimizer per lineage (OPTIMIZER_BAKEOFF_20261011.md). A LionSR full state refuses to resume under MuonSR in one sentence; a continuation switches only with `--curriculum-change latent_optimizer --optimizer-state fresh`. |
| drift monitor | none (only a quantization gate at evals) | none | accidental (code age) | the shared monitor (`train/drift_monitor.py`, the direct stage's): 48 probes, `drift_rows` 16 at the eval cadence, response KL warns 0.5 (runs the gate), stops 1.5, gate floor base − 0.1; KL reference = the model at the E2E start, captured per run; final gate in the report | Acceptance criterion 3 needs a guard and a final gate; the E2E stage had neither. A drift stop exits 3 and a resume of that run is refused. |
| latent clip | in-backward latents unclipped (Mellum) | exact global clip 1.0 (latents step after it) | not math-neutral | `latent_grad_clip` exists (in-backward latents scaled by min(1, 1.0 / previous global norm)) but is off | Measured harmful (smoke 2026-10-11, LFM direct): global norms are 20–650, so clip 1.0 is always on, and the lagged estimate mis-weights updates up to 20x; held CE after 24 updates 4.48 against 3.79 unclipped. The exact clip of the non-in-backward path is always on too; a decision for the coordinator. |
| memory_gb, graph_memory_gb, graph_headroom_gb, producer_batch_size, backward_policy, checkpoint_layers | defaults | 7.5, 6.9, 0.4, 1, auto, true | fit-only | per machine | — |
| eval_every / checkpoint_every | 0 / 25 (trainer default) | 25 / 25 at launch, 200 / 200 on resume | accidental | 200 / 200, also the trainer defaults now | Cadence policy. The trainer's own defaults were inconsistent (`--checkpoint-every` 25, consumer `eval_every` 64). |

### E2E acceptance (both lines; `line-parity.json` `stages.e2e.acceptance`)

Pop asked for this (17:58): the token-readout criterion is shared line policy, not a per-run workaround.
A checkpoint of the end-to-end stage is selected only if all of these hold on held data and the exact exported weights:

1. **Token-readout regression.**
   - Held crisp CE through the trained TokenChoiceProjection, minus held crisp CE through the raw head
     (lm_head(norm(h))).
   - Both are measured on the same weights, the same exact held rows (the saved CE selection: all held reader rows)
     and the same crisp messages and targets.
   - Evaluator: `eval.direct_token_readout`, held-consumer arm.
   - The penalty must be at most **0.05 nats**, the same noise margin as the reader controls.
   - Raw-head and TCP CE against the run's initial crisp CE are reported, which separates backbone drift from
     readout drift.
   - Evidence: Pop's native epoch at update 600 had TCP 2.094 against raw 1.270 (+0.823); at 687 it had 1.398
     against 1.262 (+0.137). The backbone-only crisp CE improved while the readout regressed, so the criterion has
     to measure the readout separately.
2. **Paired reader, values against tokens.** These are the gate of NEURALESE_DIRECT_V1.md §4–5:
   - consumer CE when reading the Neuralese block must be at most the CE when reading the gold crisp text (V ≤ T);
   - V must beat the shuffled block and the zeroed block by 0.05 nats each;
   - on exact-oracle records, V's success rate must be at least T's minus 0.02;
   - greedy-T is reported only.
3. The shared behaviour gate passes (the hard criterion: at least base minus 0.1), the drift guard never stopped
   (response KL hard stop 1.5; KL 0.5 only warns and runs the gate, decision 2026-10-11; KL is reported with the
   result), and runtime is requalified on the exact final weights.

Pop's best225 (written 0.929 against shuffled 1.359) and final687 (1.446 against 1.609) are judged by this. Neither
is selected until criterion 1 is measured on it.

### Cadence policy (both stages, both lines)

- **Evaluation:** at the stage end, plus the cheap drift monitor at a cadence that keeps all evaluation near 3% of
  wall time. Per line, cadence steps = measured eval seconds / (0.03 x measured step seconds). These values are
  fit-only and declared.
  - Mellum direct, measured on v2c: step 6.8 s, drift check about 50 s, gate 190 s. That gives `drift_every` 500
    (1.4%) and `drift_gate_every` 1500 (about 2%). `drift_gate_rise` still gates at once on a sudden rise.
  - LFM: Pop sets its values from its own seconds.
- **Checkpoints:** full state at half way and at the end, plus the final and emergency (signal) saves. No
  more-frequent checkpoints (owner).
- **E2E:** `eval_every` 200, `checkpoint_every` 200 (Pop's measured choice). These are also `trajectories.py`'s
  defaults now.

### Not yet shared

- The Mellum direct line's next recipe is `neuralese-direct-mellum-v3`. It binds the registered Mellum tokenizer twin
  (`neuralese-foundation-text-input-4962-mellum21-twin-20261011-v1`, sha 7fe4e055…; per-row ids, splits, groups and
  records equal) and Mellum's measured base gate. Its QAT component has no preserve stream and no gate of its own:
  drift reference and gate are the shared monitor.
- Mellum at the shared context, measured 2026-10-11 (`runs/neuralese-direct-line-parity-20261011/smoke32k`;
  tokens 32768, target 1024; the 24 longest documents, 11.2k–13.4k tokens; BF16; 12 updates, no full state):
  - polled ledger peak 59.6 GB;
  - 10.4 s/update mean over updates 2–12 (7.0–13.2 s), against 6.8 s at 8192 in v2c (with q4 points mixed in);
  - most documents are shorter than 8192 (median 4.5k), so their windows are unchanged. A 2000-update v3 run should
    take about 4–5 h at a claim of about 72 GB.
- Mellum v2c (V9 data, 8192/1024) completed and qualified: held CE 2.224 → 0.209, gate 20/24 against base 18/24,
  response KL 0.34. It is the last Mellum direct run on the old settings.
- Twin caveat: Mellum's template shows every earlier turn's reasoning (2921 documents, 5627 turns), which LFM's
  template drops, so Mellum has 384k supervised tokens against LFM's 344k. This is template-inherent.
- LFM E2E: Pop's running endpoint epoch v3 (`luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`) already matches the E2E base up to the declared differences and is the LFM line's checked E2E recipe. Later E2E launches on both lines declare `parity {line, stage: e2e}`.
- The running Mellum v2c finishes as is. It is recorded as trained on the V9 set (4480 documents) with 8192/1024.
