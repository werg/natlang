# core_text_warmup audit (2026-10-11)

Owner request: the Mellum run-v9 behaviour-gate failure at update 4096 (arithmetic 92% to 0%, loops, held KL 0.92)
and Pop's LFM-350M regression (math 0/12, tool use 0/12, Natlang `eval`/`return_result` on unrelated prompts, 14%
drift without Lion or QAT) suggest bugs in the warm-up stage beyond LionSR. This is a read-and-test audit of
`train/text_warmup.py` and its collaborators at origin/main 4d044692. Line numbers refer to that commit. Evidence
comes from `runs/mellum-foundation-qat-20261010/run-v9/core_text_warmup/{plan,train,eval,baseline}.json*`, from CPU
scripts run under the memory ledger, and from `tests/neuralese/test_warmup_audit.py`, which is new and passes.

No Mellum forward was run because the DGX had about 19 GB free. Every finding below comes from code, logs or
CPU checks.

## What the run log shows first

| update | phase | held CE on assistant replies (pass 0) | train preserve KL (BF16 point) |
|---|---|---|---|
| 0 to 2560 | projection only (backbone frozen) | 2.52 | not applied |
| 2564 | first backbone updates, lr scale 0.25 | 2.52 (last eval) | 0.017 |
| 2567 to 2600 | same | | 0.12, 0.60, 0.72, 0.77, 0.80 ... |
| 2816 | | 0.61 | about 0.6 |
| 4096 | lr scale 1.0, 2 passes | 0.43 | 0.78 (bucket mean) |

Within 256 backbone updates the model moved 1.9 nats per token toward the Natlang corpus. The preserve stream's KL
rose from 0.017 to about 0.7 within about 20 updates and never came back. Whatever the optimizer, the objective and
data pull hard toward the corpus, and the anchor pulling back is weak (findings 1 to 4).

## Ranked findings

### 1. BUG (Mellum line): the system prompt that conditions Natlang behaviour is never in a training window

- Run-v9 uses `--tokens 1024 --prefix-tokens 32` with no `--target-tokens` (plan.json). The rendered native
  documents average 5,003 tokens, of which the leading system prompt averages 4,206 tokens. `--mask-system-prompt`
  makes those tokens context-only (`prepare_text_windows`, text_warmup.py:930-935; `document_windows` :814-821).
- Without `target_tokens`, each window carries only `prefix_tokens` (32) of preceding context
  (`context_capacity=prefix_tokens`, :812-813). Every offset-0 window lies wholly inside the system prompt and is
  dropped (`len(chunk)<=width`).
- Measured on 600 training documents with the real Mellum tokenizer
  (`/home/werg/.claude/jobs/9c849f00/tmp/audit/window_stats.py`): 1,255 windows and 297,133 supervised assistant
  tokens.
  - 0.0% of supervised assistant tokens have the `<|im_start|>system` header in their window.
  - 0.0% have the full system prompt in their window.
  - 55% have no system text at all. The other 45% see only a tail fragment of the TypeScript API declarations.
  - No window starts at the document start.
- Consequence: the model learns that after `<|im_start|>assistant` it emits a `<tool_call>` with Natlang `eval`
  code. It learns this unconditionally, because the prompt that should gate the behaviour is invisible. This
  matches "emits Natlang eval/return_result on unrelated prompts", and it is the opposite of what system masking
  intended ("context only", not "absent").
- Pop's V8 uses a 32K context, so this finding is Mellum-specific. Pop's symptom has other causes (findings 2
  to 4).
- Fix: run the stage with bounded targets and full context. The trainer already supports this:
  `--target-tokens 1024 --tokens 8192` (or larger than the longest system prompt plus the target), so each window
  keeps all preceding document tokens up to `tokens` (:807-810). The memory forecast grows with context, so check
  the preflight. If memory forbids that, pin the system prompt as a crisp prefix of every window. Add a gate to the
  window builder: refuse, or log, when the fraction of supervised tokens whose window lacks the document's
  conditioning prefix is above zero.

### 2. BUG (both lines): behaviour preservation exists only when the stage has a quantization component

- `preserve` and the behaviour gate are built only inside `if quant is not None:` (text_warmup.py:1476-1485). The
  drift guard `behaviour_drift` also needs `preserve.held` (:2224-2229).
- A BF16 lineage, such as Pop's LFM-350M with no QAT, therefore trains with no anchor to the base model's ordinary
  behaviour and no drift metric. Its only regularizer is the text-weight CE on the same corpus.
- Fix: make `preserve`, `gate` and drift a stage-level component that is independent of the precision points. For
  LFM, register a self-distilled teacher top-k artifact of LFM's own chat/math/tool responses (same format as
  `mellum21-teacher-v3-top64-20261010`).

### 3. DATA (both lines): the corpus teaches unconditional Natlang tool calls, and the anchor is too weak to resist

- The assembled-v3 native corpus, rendered with the Mellum tokenizer (`audit/data_stats.py`, 800 sampled
  documents), contains no ordinary-behaviour examples:
  - 4,480 documents (2,776 train), one cohort (`native`). The harness cohort was not admitted.
  - Every document contains a tool call.
  - 85.4% of assistant-reply tokens sit inside `<tool_call>` JSON (Natlang `eval` code and `return_result`).
  - Only 8 assistant turns in 800 documents have neither a tool call nor a think block.
  - No plain chat and no math answers.
  - Supervised token shares after system masking: assistant reply 46.7%, tool output 29.4% (weight 0.25 in the
    context half), user 16.0%, reasoning 7.3%.
- Objective balance per update (`perform_update`, :2501-2519; `objective_pass`, :1681-1685):
  - Main objective: the mean over passes of (projection MSE + input-map MSE + 1.25 × CE for pass 0, or 1.0 × CE for
    pass 1), on 2 × about 992 target tokens.
  - Preserve: 0.5 × KL on one teacher record of up to 4,096 tokens.
  - Both are scaled by the same sampled precision weight.
  - Preserve is a single record against a dense CE pull toward one format. The measured train preserve KL climbs
    from 0.017 to about 0.8 regardless.
- Fixes:
  - Add an ordinary-behaviour stream to the text data: the base model's own answers to chat, math and generic
    tool-use prompts, rendered with and without a Natlang system prompt, as self-distillation targets.
  - Or raise preserve to several records per update, so its token count is comparable to the main batch.
  - Add negative contexts: the same user prompts without the Natlang system prompt, answered plainly.
  - Owner policy (breadth via natlang domains, no anchors capping learning) bears on the choice. The finding is
    that the stage currently has zero examples of when not to call `eval`.

### 4. DESIGN RISK (both lines): the full-projection embedding loss trains the backbone's final state, and its head got worse while training

- The full-projection error flows into `top` (the backbone's last hidden state), undetached, at
  `--embedding-weight 1.0` alongside CE (`projection_errors`, :481-486; loss :1681-1683).
- Its forward is `E[argmax(logits(h))] + proj(norm(h))`. Its backward is a straight-through softmax surrogate
  (`causal_feedback.py:117-125`), so the term amounts to an argmax-accuracy objective on the backbone. This
  sharpens toward the corpus mode, which is a plausible contributor to greedy loops.
- During the 2,560 projection-only updates, the head (`heads.content.proj`, zero-initialised, Muon at
  `sketch_lr` 3e-4) made its own objective worse:
  - Training relative MSE went from 2.36 (first 128 updates) to about 3.0 (updates 2,048 to 2,560).
  - Held pass-0 relative MSE went from 1.758 (step-0 baseline, zero head) to 2.08-2.20.
  - The schedule declared the plateau as "stale" (plateau eval 4, best 1.79 = the untrained head) and started
    backbone training with this head in place.
  - In the first 128 backbone updates the training projection error fell from about 3.0 to 0.99 and accuracy rose
    from 0.52 to 0.71. The backbone was bent to serve a head that never learned anything useful.
- A CPU simulation (`audit/proj_head_sim.py`) of a zero-init linear head on a weak-signal residual under
  Muon/AdamW at 3e-4 with no decay reproduces the pattern on a smaller scale: held relative MSE never improves on
  the zero init (0.99 to 0.97-1.08).
- Fixes:
  - Detach `top`, or scale its gradient (for example `scale_gradient(top, 0.05)`, as `gold_completion` already
    does for its auxiliary), for the full-projection term during warm-up.
  - When the projection-first phase ends without beating its initial error, restore the best (here the initial)
    head instead of carrying the drifted one into backbone training.
  - Lower or decay the head rate.

### 5. DESIGN RISK (both lines): pass 1 trains the whole backbone on read-adapter inputs at full CE weight

- With `sequence_passes >= 2` (from update 3,073 in run-v9), pass 1 feeds
  `read_embeddings(input_map(E[gold]))` for every span position (`mapped_completions`, :548-553). Pop's fixup
  feeds its AR projected history instead (:598-601). Pass 1's CE back-propagates into every backbone layer at
  weight 1.0, half the CE of an update.
- The read adapter (`heads.NeuraleseReadAdapter`) exists to absorb reader-side adaptation. The backbone still
  adapts its lower layers to non-token inputs, which shifts plain-token processing, because there is one set of
  weights.
- The plain path itself is correct:
  - Pass 0 uses `forward_ids`, which embeds plain tokens (only the two marker rows are replaced; `lfm2_port.py:387-391`).
  - Preserve and the gate use `backbone.hf` directly (`quantization.py:420-446`; `weighted_topk_kl` calls
    `model.model(input_ids=...)`).
  - So the preserve stream is on the plain token path, as required.
- Fix to measure: put the pass-1 gradient on the read adapter and heads only, or scale it into the backbone (for
  example 0.1). Compare held KL against the plain path after the same number of updates.

### 6. KNOWN, owned by the bug hunt: fixed-size latent steps; plus the router under LionSR

- A CPU simulation (`audit/sr_sim.py`) runs the real `optim._lion_rows` on a 256 × 4096 BF16 latent with random,
  uninformative signs. Each step moves `lr x row_scale` (4.45e-4 of |w| at lr 3e-4):
  - lr 3e-4, 1,000 steps: 8.1% relative drift. The same signs in FP32 give 7.1%. Stochastic rounding alone adds
    3.8%.
  - lr 7.5e-5, 1,000 steps: 2.7%.
  - Drift therefore barely depends on the gradient. That matches the "uniform 11%" drift and the bug hunt's
    preserve-only null test.
- Constant lr after the 4-eval ramp, with no decay (text_warmup.py:2889-2890; `foundation_schedule.py:117`).
- The router (`.mlp.gate.weight`) is selected by the latent policy (`backbone_policy.py:71-73`). It then enters the
  LionSR partition, because every backbone tensor with `ndim>=2` goes there (text_warmup.py:1549-1550), with a
  per-row ternary-scale step. `muon_eligible` explicitly routes MoE routers away from matrix optimizers
  (`optim.py:31-32`), and the latent path bypasses that rule. Every router step reroutes tokens.
- Fix: keep routers (and any non-hidden-state matrix) on AdamW at the backbone lr, or freeze them. Decay the latent
  lr. MuonSR (4d044692) already skips idle experts.

### 7. DESIGN RISK (Mellum): BF16 behaviour is protected only in BF16-sampled updates

- In `sample` mode one point is drawn per update with probability proportional to its weight (`quantization.py:294-306`).
  At update 4,096: BF16 0.40, q4 at λ=1 0.20, ternary-experts at λ=0.125 0.40.
- The preserve KL is computed inside the same precision context (text_warmup.py:2503-2519). In 60% of late updates,
  then, the BF16 latents take a full LionSR step chosen to fix the quantized model, and its preserve term
  protects the quantized forward, not BF16.
- The STE trains the latents to be good under quantization; nothing in those updates holds the BF16 forward.
- Fix: in every update, add the preserve term at BF16 regardless of the sampled point, or add a BF16 KL-to-teacher
  term on the main batch. Note that `--latent-step-in-backward` requires one combined backward: add the BF16
  preserve graph to the same `terms` list.

### 8. MINOR: the preserve KL pushes the student's mass outside the teacher's top 64 to zero

- `weighted_topk_kl` (`maple/qat_convert.py:235-265`) computes forward KL(teacher || student), the right direction:
  - temperature 1;
  - teacher renormalised over its top 64;
  - student at full log-softmax, not renormalised.
- The term therefore includes −log(student mass on the teacher's top 64), so its minimum is a student with all its
  mass on those 64 tokens. That is a mild sharpening; the step-0 BF16 value is 0.017.
- The teacher shards are for the same model and tokenizer: `/home/werg/data/mellum-qat/teacher-v3/teacher.json`
  has model `mellum21-12b-a2.5b-thinking`, k 64, Mellum's self-distill v3 records.
- Pinned by `test_preserve_kl_is_forward_kl_to_the_renormalised_teacher_top_k`.
- Option: add a residual bucket (teacher residual mass versus student residual mass) so the KL has its true minimum
  at the teacher.

### 9. MINOR: batch companions duplicate unique-shape windows

- `DocumentWindowSampler.batch` (`text_supervision.py:118-124`) draws companions with the same `(prefix, length)`.
- The windows that straddle the end of the system prompt have unique prefix widths, so the companion is usually the
  same window again.
- Those windows hold the first user turn and the first assistant turn, the most behaviour-defining ones, so they
  get twice the weight in their updates.
- Fix: pad or crop to shared shapes, or accept a smaller batch.

### 10. LOW: window position 0 is an arbitrary mid-document token

- Mellum's chat template starts with `<|im_start|>` and has no BOS. Every training window starts mid-document (see
  finding 1). This is ordinary packing practice, so the risk is low, but the attention-sink token differs from
  inference.
- Finding 1's fix removes this too.

## Checked and fine

- **Target alignment.** `top[:, j]` predicts `span[:, j]` in both passes. Pass 0 starts at `prefix-1`. Pass 1
  reads mapped slots `span[:, :-1]` through a left-padded causal conv (`input_map.py:33-34`). Changing `span[:, k]`
  leaves `top[:, :k+1]` unchanged (`test_both_sequence_passes_predict_each_target_without_reading_it`). Readout CE
  is a weighted token mean (`chunked_readout`, :475-478). Position weights are normalised per window to mean 1
  (`text_supervision.py:52`).
- **Padding.** There is none: batches are shape-matched. Each document is its own windows; there is no packing
  across documents.
- **System masking.** Only the leading system message is masked (:930-935), as declared. Tool responses are not
  masked: their weight is 0.25 in the context half (`balanced_position_weights`), and the role is detected through
  the real token sequence `[233, 31]` after `user`. User turns are supervised at weight 1.0 (owner decision on
  whole-trajectory supervision).
- **Frozen parts.** The latent policy trains `model.layers.*` attention projections, experts, routers and layer
  norms only (`backbone_policy.py:71-75`). `embed_tokens`, `lm_head` and the final norm stay frozen. Marker rows
  are not in the trainable set.
- **Weight decay** is 0 everywhere: LionSR and MuonSR groups (`optim.py:68,71`), Muon and AdamW children
  (`optim.py:76,78`), and the latent and AdamW groups in text_warmup (:1566,1570).
- **Clipping.** The global norm is clipped to 1.0, including the norm of the latents already consumed
  (`trajectory_state.py:26-39`). The in-backward latents themselves are not clipped (declared). Typical norms of
  17 to 36 give clip coefficients of 0.03 to 0.06 on heads and norms. Under AdamW and Muon this is nearly
  scale-free.
- **Sum versus mean.** Passes are averaged (`loss/passes`). Preserve is added once per update and not divided by
  passes. One precision point is used per update (sample mode, enforced with in-backward, :2499-2500). Its loss is
  scaled by the total active weight: 1.0, then 1.5, then 2.5 as points ramp in. This scale cancels under
  sign/orthogonalised/Adam updates and changes only the clip ratio. "Precision weight 1.5" is that total, not a
  sum of several precision terms.
- **LoRA and effective weights (item 7).** The Mellum path has no LoRA:
  - heads backbone identity `precision bf16`, `init: bf16` (convert-v3 latents declined);
  - no `lora`, `backbone_trainables` or `maple_qat` in `runtime_qualification/heads.pt`;
  - the latent policy injects none.
  - The gate at update 4,096 ran in-process on the live weights, so restore or merge double-application cannot
    explain the Mellum failure.
  - Resume copies saved latents into the live parameters by name after the parametrizations are installed (:1478,
    :1820). That is not a merge.
  - Pop's LFM LoRA restore (`serve/__init__.py:88-104`, `recurrence_checkpoint.py:34-50`) was not exercised here.

## Suggested order of fixes

1. Bounded targets with full context (finding 1). This is a recipe change only.
2. Preserve and drift as a stage component on every lineage, including BF16-only lineages (finding 2), plus an
   ordinary-behaviour self-distillation stream (finding 3).
3. Detach or scale the full-projection gradient into the backbone, and restore the best head at the end of the
   projection-first phase (finding 4).
4. Run a BF16 preserve term in every update (finding 7). Take routers off the latent sign optimizer, and decay the
   latent lr (finding 6, together with the bug hunt's MuonSR).
5. Measure a pass-1 gradient scale into the backbone (finding 5) against the held KL drift metric.

Audit scripts, kept outside the repository: `/home/werg/.claude/jobs/9c849f00/tmp/audit/{data_stats,window_stats,proj_head_sim,sr_sim}.py`.
