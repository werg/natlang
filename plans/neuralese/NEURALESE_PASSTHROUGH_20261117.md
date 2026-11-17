# Neuralese as a differentiable pass-through in the learners (2026-11-17)

Owner, 2026-11-17: "the whole idea is to use them as differentiable pass-through." A consumer call's imported Neuralese
state is computed in-graph from its producer's tokens, so the consumer's loss reaches the weights that computed the
state, through the real producer->consumer dependency. The learner no longer reads stored KV states.

## What the actor actually does today (verified on grpo-v3/run1 rollouts and its actor store)

- **Producers write text; the host writes the state.** Every native state in today's rollouts is a **text write**
  (`execution_graph` event `kind: block_write`, `source_kind: typed-text-result`, `source: return_result`). The producer
  invocation returns its value as ordinary sampled text (all its samples have `input_role: ordinary`). The runtime then
  calls `native_write_text`. FALLOW C6 records that native writing (the model sampling a produce span itself) is off.
- **The write is a standalone document.** `sglang_actor._sglang_write_text` (sglang_actor.py:300-327) and
  `engine._native_write_text_impl` (engine.py:1179-1209) both run `[produce_open] + template_tokens(text)` from
  position 0. There is no prompt context and no BOS. They capture rows 1..n (the close marker is not part of it).
  - In the fork (`../sglang-natlang` `models/lfm3.py` 560-590 and 850-867, `NatlangCapture` 1016-1075), a row's state
    is the K after the K norm and before RoPE, V, and the conv input `B*x`. It is stored as FP32 of the BF16 values.
  - Inside the span the input embedding is the export's producer table, `natlang_producer_embed` (lfm3.py 1338-1395).
- **The write is recorded in the rollout.** The `block_write` event carries `source_capture`: `segments` (token ids
  and producing flags), `payload_start`, `payload_stop` and the policy identity. This is the exact input of the
  capture, and it is archived with every rollout (`ops/*/rollouts/g*.rollouts.jsonl.zst`).
  - A consumer prompt names the block by `{kind: native_state, id, source_occurrence}` (`source_occurrence` is the
    `block_write` node). The import mapping therefore needs no actor store.
  - The store is collected during runs (`grpo.collect_blocks`). The 449 own Neuralese successes have lost their
    stores, but not their `source_capture`s.
- **The import.** The consumer gets `consume_open`, then `count` placeholder positions, then `consume_close`. At the
  placeholder rows every layer's pre-RoPE K, V and conv `B*x` are the stored ones (`NatlangImport`, lfm3.py 1080-1150).
  RoPE is applied at the consumer's positions.
- **The producer table is an identity map today.** For the start policy 5ff07638, `natlang_producer_embed` equals the
  merged `embed_tokens` bit for bit. The checkpoint's `interface_state` has no producer adapter, only `input_rows` and
  `extra_input_rows`.
  - **Defect found:** `grpo.WeightSync` (grpo.py:653-700) keeps the producer table at the export's start values, while
    `embed_tokens` trains (grpo.py:1666-1669, 2400, 2419). From step 1 on, the actor's text writes embed with the start
    embedding while ordinary tokens use the trained one.
  - Under native semantics (`producer = adapter(embed)`, with the adapter the identity here) the table must follow
    `embed_tokens`. The Python native engine computes it from live weights; the GGUF and HF exports compute it at
    export (native_hf.py:91-99, 129).

## Inventory: every component of the Neuralese state handoff

| Component | Where | Changes? | Why / citation |
|---|---|---|---|
| SGLang fork capture/import | `../sglang-natlang` `models/lfm3.py` 560-590, 850-867, 1016-1150, 1338-1395 | no | The learner reproduces its arithmetic (same span, positions, K-norm/pre-RoPE point, `B*x`, producer table). Nothing in the pass-through needs a new serving feature. |
| SGLang front | `serve/sglang_actor.py` 300-327 (text write), 1036-1100 (import), 1153-1215 (span capture) | no | Already records `source_capture` for every write (through the runtime's `block_write` event). |
| SGLang in-place weight sync | `grpo.WeightSync` grpo.py:653-760 | **yes** | Publishes `model.natlang_producer_embed` from the trained embedding (identity producer adapter) instead of the frozen start table. Opt-out: `--frozen-producer-table`. |
| Python native engine | `serve/engine.py` 1179-1209, `model/shared_state.py` 1020, 1086 | no | Computes the producer embedding from its live weights (`adapter(embed)`): already consistent. |
| Managed llama-server, in-page wllama | GGUF from an export (docs/ENGINE_PARITY.md 85-91) | no | Serving only. Their producer adapter is baked at export from the same checkpoint, so they are consistent; a learner-side lowering has no serving counterpart. |
| Fast learner kernels | `lfm3_fast_train.conv_layer` / `attention_layer` (106-107, 130-132), `hidden` (371-403) | **yes** | A new in-graph import, `pass`: at the consumer's import rows, K/V/`B*x` = the same layer's values at the producer document's span rows (a differentiable gather in the same packed forward), or `.detach()`ed with the opt-out. The fixed `import_tensors` path stays as a fallback for blocks with no recorded source. |
| Prefix-tree path | `lfm3_fast_train._hidden_past`, `run_prefix_tree` | no | Chains that import stay whole (grpo.py:2196-2200; `hidden` refuses `past` + imports, 380-381); their producer documents ride in the same whole forward. |
| Stock HF path | `exp0_sft.hidden` (122-155) | no (refuses `pass`) | Parity reference only. It is the independent reference the tests use to capture a producer state. |
| Lowering of rollouts | `lower_rollout_turns.prompt_imports` / `episode_turns` / `decision_turns` (53-155) | **yes** | Each import is resolved to its `block_write` `source_capture` from the row's `execution_graph`: the producer document (ids up to `payload_stop`) and its span. Nested `native_state` segments inside a source are resolved recursively. The store is read only for blocks without a source (fixed fallback, counted). |
| Chain building | `exp0_sft.chains_of` (53-86) | **yes** | Carries `sources` ({block_id: document}) with `imports`. |
| SFT learner | `exp0_sft.packed_rows` / `row_tensors` / `train` / `validation_loss` (219-246, 322-470) | **yes** | The refusal at 224 goes. Each packed row appends its producer documents and passes `pass` imports. Opt-out: `--no-neuralese-passthrough` (detach). |
| GRPO learner | `grpo.chain_inputs` (2218-2224), `decision_tensors` / `decision_inputs` (1206-1225, 2236-2245), the reference pass | **yes** | Same packing. The KL reference recomputes the state under the reference weights (before, it used the actor's state). Opt-out: `--no-neuralese-passthrough`. |
| Behaviour probe | `behaviour_probe.py` 28-50 | no | It selects only calls without imports (`call['imports']` excluded at line 50). |
| Held eval | `rl/held_eval.py` | no | Actor-only measurement; no learner forward. |
| Rollout archives | `grpo.archive_rollouts` (597-613) | no | They already keep `execution_graph` with `source_capture`, the only input the pass-through needs. |
| SFT data | `sft_sources.py` | **yes** | A new `neuralese-graph` subcommand: whole Neuralese episodes from the exact actor captures of the original rollouts (every call, post-import consumers included) with their sources. |
| FALLOW-22 graphs | `/mnt/external/natlang-fallow22-20261110/` | not wired | Stale: their system prompt is 33.4k characters against today's 23.2k (bird-s0 n0 vs grpo-v3/run1). They are teacher-program chains with native-write spans inside producer calls, a form today's runtime does not produce. The lowering's source form (a span inside another node) covers them if they are ever re-rendered. |

## Design: training-side lowering equal to the actor's capture

1. **Source of truth.** For an import `(row r, block b, count n)` of a consumer call, the producer document is
   `ids = concat(segment.token_ids for segments)[:payload_stop]` from `b`'s `source_capture`, with span rows
   `[payload_start, payload_stop)`.
   - For a text write: `[produce_open] + template_tokens(text)` and rows 1..n, exactly the capture request's ids and
     span.
   - The placeholder tokens of the consumer prompt are `ids[payload_start:payload_stop]` (= `block.token_ids`), as
     before.
2. **One packed forward.** The consumer's sequence (a chain or a packed row) is followed by its distinct producer
   documents as separate documents (`lengths`: positions restart at 0, attention and conv stay within the document).
   This gives the actor's positions 0..n with no BOS.
3. **The gather.** In every layer, after the K norm and before RoPE:
   - `k = k.index_put((rows,), k[source])` and the same for `v`;
   - in conv layers, `bx = bx.index_put((rows,), bx[source])`;
   - `rows` are the consumer's import rows and `source` the producer document's span rows.

   Source rows are produced tokens, never import rows (checked), so a DAG of imports (a producer that itself imports)
   resolves in one forward: every layer reads the producer's own values of that layer. RoPE is applied afterwards at
   the destination rows, so the keys rotate at the consumer's positions as in `NatlangImport`.
4. **Producer embedding.** The checkpoint has no producer adapter, so producing rows embed with `embed_tokens`, which
   equals the actor's table at sync time once WeightSync publishes it (above). The learner refuses a checkpoint whose
   `interface_state` carries a producer adapter (a non-identity table) until that is modelled.
5. **Gradient.** The consumer's loss reaches every weight that computed the producer's span state: the layers below
   each layer, the embeddings of the producer's tokens, and the norms. The producer's sampled text is discrete, so no
   gradient flows into its sampling. That text trains as the producer's own call (its policy term or SFT target), as
   before.
   - Opt-out `--no-neuralese-passthrough`: the same in-graph values, `.detach()`ed. This is a fixed context that equals
     the old stored-state path at the BF16 floor without needing the store.
6. **Off-policy-ness.** The state is a deterministic function of the weights and the producer's tokens. With
   pass-through the learner's state is the current policy's, and `rho` compares the consumer's tokens under the
   learner's state against the actor's. That is the correct ratio for a deterministic intermediate; at version 0 the
   two states agree at the BF16 floor.
7. **Cost.** A text-write producer document is the value text plus one token: 8-91 tokens in the samples checked,
   tiny beside 1-30k-token consumer chains. A block imported by several calls of one chain is computed once per
   forward. Activations add `doc_tokens x 85 KB x layers`.

## Tests and measurements

- CPU, tiny LFM3:
  - the in-graph pass equals the stock-HF capture of the producer document spliced through `import_tensors`;
  - an autograd gradient check: the consumer's loss gradient reaches the producer tokens' embedding rows, and a
    float64 finite difference on a layer weight agrees with autograd, through the pass;
  - detach gives zero gradient there.
- GPU, ledger, about 8 GB: real version-0 blocks from grpo-v3/run1's actor store (written by the 5ff07638 export)
  against the learner's in-graph state of the same `source_capture`. The floor is learner-fast vs stock HF on the same
  document.
- Bench, ledger, ≤15 GB: forward+backward tokens/s and CUDA peak on Neuralese chains, pass vs detach.

## Results (2026-11-17)

Commits:
- cdb97e649: WeightSync publishes the producer table from the trained embedding. Opt-out `--frozen-producer-table`; a
  checkpoint with a producer adapter is refused.
- 993df3f3d: the pass-through in `lfm3_fast_train` / `lower_rollout_turns` / `exp0_sft` / `grpo`, on by default.
  Opt-out `--no-neuralese-passthrough` in both learners.
- The capture checks and the bench (`scripts/neuralese_data/passthrough_check.py`) and the SFT data step
  (`sft_sources.py neuralese-graph`) follow in later commits.

**CPU tests** (`tests/neuralese/test_lfm3_fast_train.py`, tiny LFM3, FP32/FP64):
- The in-graph state equals an independent capture of the document: stock HF hooks, spliced as the fork does. The
  consumer's hidden states agree within 1e-4.
- The consumer's loss reaches the embedding rows of tokens that occur only in the producer document; frozen, it does
  not. The frozen gradients equal the stored-state path's.
- On weights acting on the producer, float64 central differences with the state recomputed equal autograd within 2e-3
  relative, and the frozen gradient differs from them.
- exp0_sft rows, steps and the prefix-tree split, and the lowering from `execution_graph` (`stop_gradient` imports
  frozen), are also covered.

**Real captures.** These are grpo-v3/run1 actor-store blocks: text writes of 1-169 tokens served by export ba7a8fed
(version 10). Because the start-version stores had already been collected, the learner used checkpoint step-00010.
1. *Per-layer state:* the relative L2 of each layer's K, V and B*x over the span.
   - Layers 0-2 agree at the BF16 floor (actor vs learner-BF16 0.0006-0.008; learner BF16 vs FP32 0.003-0.008).
   - From layer 3 on, the first consumer of an attention output, the actor deviates from FP32 about 2.5-3x more than the
     learner's BF16 does. For the 32-169-token blocks, the median over layers is: actor vs FP32 0.048-0.055, learner
     BF16 vs FP32 0.017-0.020, actor vs learner 0.047-0.056.
   - Embedding the span rows with the actor's stale producer table instead of the trained embedding changes nothing
     measurable at version 10. Ten steps of embedding updates are far below this noise.
   - Reading: the learner's state is the BF16 state. The actor adds error that starts at its first attention read.
     That is consistent with its FP8 KV cache (an attested engine choice), but **not verified**: a BF16-KV actor
     capture needs an SGLang unit of about 24 GB.
2. *End to end* (6 consumer calls, 628-2,612 sampled tokens each, the actor's recorded log-probabilities). Mean
   |dlogp| per token:

   | comparison | mean abs dlogp |
   |---|---|
   | learner with in-graph state vs learner with the actor's stored block | 0.013-0.019 |
   | in-graph vs actor | 0.030-0.044 |
   | stored block vs actor (today's learner) | 0.032-0.044 |
   | no state (placeholders as plain tokens) vs actor | 0.066-0.151 |

   The in-graph state predicts the actor's tokens as well as the actor's own stored state does.

**Bench.** Forward+backward, BF16, activation budget 3 GB, 6 importing chains of 5.1-7.5k tokens with 10-97-token
producer documents, steady-state repeats. The DGX GPU was shared with the running SFT, so the absolute rates are low.

| mode | tok/s | CUDA peak |
|---|---|---|
| no producer documents (the stored-state path's compute) | 2,204 | 9.21 GB |
| pass-through | 2,206 | 9.24 GB |
| frozen | 2,171 | 9.24 GB |

The pass-through costs nothing measurable: the documents are 0.2-1.9% of the tokens.

**Side finding, reported, not investigated.**
- Compiled `lfm3_fast_train` layers differ from the eager kernels by mean |dlogp| 0.037 (max 0.62) in no-grad mode, and
  0.009 in grad mode, on a chain **without** imports. They also differ between processes.
- Eager is deterministic, and there the pass-through and frozen modes are bit-identical. The differing pass/frozen
  losses in the compiled bench are therefore this compile noise, not the pass-through.

**SFT data.** `sft_sources.py neuralese-graph` (see `build.json` of `runs/sft-neuralese-graph-20261117-v1`) lowers the
own Neuralese successes whole from the archived rollouts, consumer calls after imports included, with their recorded
producer documents.

**FALLOW-22:** stale (old surface, native-write spans), not wired.

**Not checked:**
- a version-0 actor capture (the start policy's store was collected);
- a BF16-KV actor capture (the FP8 hypothesis);
- the SGLang import of a block against `import_tensors` beyond the existing parity tests;
- a GRPO or exp0_sft optimizer step with pass-through on the DGX: the bench ran forward+backward only, no AdamW, because
  of the 15 GB limit;
- held-out effect;
- a native-write span (inside a producer call): none exist today, and `write_sources` refuses sources that import
  state themselves (counted).
