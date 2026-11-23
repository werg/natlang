# Neuralese read-back: static investigation (2026-11-23)

The observation, from `plans/neuralese/HELD_FAILURES_20261123.md` finding 3.1: `natlang.read` (`natlang:model-read/1`)
almost never returns the text its producer wrote into a `Neuralese<…>` value. Exact matches were 1 of 87 in v7, 1 of 87
in v9 and 0 of 24 in v6h4 (Pop SGLang), and 13 of 465 in aei1's DGX collection. A two-line string came back as `jinja2`,
although the request carried the state (`native_state.imports` mode `exact-native-state`).

This was a second, independent investigation. It used only code reading and recorded data: no model ran, and no
forward pass, live engine or GPU was used. It covers the code at the episodes' pins: natlang 5b58f944 for the held runs,
d29451b8 for aei1, fork dbc51ddb72 on Pop and ec27f0809a on the DGX.

**Result.** The data path delivers the producer's state to the reader. In the code and records checked, no place loses,
misplaces or detaches it. The reader does use the state, but only weakly: content that exists only in the state does
reach some outputs. The weakness has three causes, ranked in §3:
- an untrained marker interface;
- a context-free producer state that the reader was barely trained to read;
- temperature-1 sampling of an exact copy, which compounds with length.

None of this is a "not a bug" verdict: §5 lists what static work could not check.

## 1. The data path, as the episodes ran it

| Step | Code (pin 5b58f944 unless noted) | Checked against the records |
|---|---|---|
| 1. The producer returns a plain string for a `Neuralese<string>` result. | `ts-host/src/native/agent.ts` decodeCall (~1030): `source_kind: typed-text-result`, `marker_context: return-result` → `decodeTurnValue` → the write port | held row-108 v9: `block_write` events with `source: return_result`, `length: 14` |
| 2. The configured writer is `write-text`: the text is forced as `[produce_open] + text`, from position 0, in an empty document (no BOS, no call context). | `ts-host/src/model/native-state-client.ts` `nativeStateWritePort` → `/v1/neuralese/native-states/write-text`; `serve/sglang_actor.py:655` `_sglang_write_text`; native twin `serve/engine.py:1262` | aei1 `block_write.source_capture`: segments `[produce_open ×1][text ×n, producing][produce_close ×1]`, payload `[1, n+1)`. Equal texts give one content id (held row-108: calls /1, /2, /3, /5 and /7 all produced `nzs1_ud5t…`). |
| 3. The fork recomputes the span and writes the state. Producing tokens are fed as `t + vocab` and read `natlang_producer_embed`. Attention layers store the normed pre-RoPE K and V; conv layers store B·x. | fork `models/lfm3.py` `NatlangCapture` (1016–1075), `Lfm3Attention.forward` (585–590), `Lfm3ShortConv.forward` (865–867), `Lfm3Model.forward` (1385–1397); `schedule_batch.natlang_producer_ids` | Stored block `nzs1_7olm…` (aei1 store): `token_ids` equal the text's ids; `positions` are 1..17; all 24 layers are finite; the stored `embeddings` equal `embed_tokens` rows bitwise. **The layer-0 B·x recomputed on the CPU from those embeddings matches the stored layer 0 at cosine 0.9997 per row** (2% relative, the vendor BF16 projection), so the captured layer and rows line up with the tokens. `natlang_producer_embed` equals `embed_tokens` bitwise for the span tokens, so the producer adapter is the identity (§3.4). |
| 4. ts-host carries the value as `{$neuralese: {id, source_occurrence}}` and passes it as `read(value)`'s argument. | `ts-host/src/native/runtime.ts` | The reader's `invocation.inputs` names the producer's block. `block_read` positions `[3, 1]` mean message 3, part 1: the tool message part. |
| 5. The reader prompt is `value: Neuralese<T> = <\|neuralese_consume\|>` + state + `<\|/neuralese_consume\|>` + `Declared value for the rest of this call.` A template readout (`template.value = 'decode'`) then forces `</think><tool_call>…<parameter=value>\n` and samples the value. | `sglang_actor._sglang_prompt`/`_sglang_import` (1461–1521), `_sglang_template_prefix` (1173), `serve/chat.py` `call_reply` (756) | The aei1 rollouts were decoded with the tokenizer (`sampling.prompt` segments). The text is as shown, the state segment sits between the two consume markers, and the forced lead is 17 tokens. The sampling law is `temperature 1, top_p 0.95`. The held request (`provider_exchange` row-108 /6) has the same shape. |
| 6. The import: `count = len(block.token_ids)` placeholder ids, hashed from (block id, index) and ≥ 2·vocab. The fork replaces K and V (after the K norm, before RoPE) in every attention layer, and B·x in every conv layer (B forced to 1), at those rows. Later tokens rotate the imported K at their own positions. | `sglang_actor._sglang_placeholders` (707); fork `NatlangImport` (1080–1150), attention (576–583), conv (858–864) | The response reports `imports[].mode = exact-native-state` and `observed_native_state_id` equal to the producer's id. |

Learner parity: the pass-through import (`lfm3_fast_train.passthrough_plan`, `attention_layer`/`conv_layer` index_put
before RoPE, `exp0_sft.row_inputs`) runs the same `[produce_open]+text` document from position 0 and splices the same
tensors. The producer table equals the embedding, so the plain `embed_tokens` lookup in `hidden` is exact. The native
engine's `splice_block` does the same (`model/shared_state.py:1401`). All three implementations agree.

## 2. Evidence about what the reader sees

aei1 DGX collection (`ops/ei/aei1/collect/v0000-e15b20a86f1c`, 710 reads; 618 with exactly one state in the prompt were
scored against the producer's `source_capture` text, whitespace-insensitive):

| State tokens | Reads | Exact | Mean similarity |
|---|---|---|---|
| < 8 | 100 | 16 | 0.40 |
| 8–15 | 166 | 1 | 0.37 |
| 16–31 | 130 | 2 | 0.32 |
| 32–63 | 111 | 0 | 0.10 |
| ≥ 64 | 111 | 0 | 0.10 |

- **The state is used, but weakly.** Take the words of the source that do not occur anywhere in the reader's text
  prompt: they could only reach the output through the state. 38 of 535 reads reproduce at least one of them. Examples:
  "Franz Xaver Gruber", "Francis of Assisi", "Quinn", "Larks", "Lorelei it seemed had recovered…". So the import is not
  missing or detached. A wholly lost import would give 0.
- **The output usually follows the declared type and drops the content.** Examples:
  - `{"party":"Quinn","size":2,"time":"18:00"}` read back as `{"party": "", "size": 0, "time": ""}` and as
    `{"party":"Quinn","size":6,"time":"2025-08-12"}`;
  - `{"kind":"outdoor"}` read back as `{"kind": "string"}`;
  - `{"cancel":false,"classes":4,"shop":0}` read back as `classes: 0` three times.
- **The distribution is flat.** Held v9 row-108 read the same block (`nzs1_ud5t…`, "Register after 0 notices.\n
  Application PL/26/0519.") three times, from identical prompts that differed only by seed. It got "Audit group
  members\nJournal entry\nTrue", then `jinja2`, then a README path with hashes.
- **The same holds on Pop (BF16 weights, fork dbc51ddb72) and the DGX (FP8 weights, fork ec27f0809a).** Weight
  precision and the fork difference are not the cause.

## 3. Where content is lost or made unreadable, ranked by evidence

### 3.1 The role-marker rows are untrained (strong evidence; it is a defect)

In the served export of the current policy (`aei1/collect/v0000-e15b20a86f1c/actor/sglang-hf/model.safetensors`, policy
e15b20a8), the merged embedding rows of `<|neuralese_produce|>`, `<|/neuralese_produce|>`, `<|neuralese_consume|>` and
`<|/neuralese_consume|>` (125033–125036):
- have norm 0.433, where an ordinary token's norm is 1.07 ± 0.11;
- have cosine ≥ 0.99996 with each other and with every unused reserved row (125029–125044). They differ from those rows
  by about 0.8% of their norm.

They are the base model's reserved-token initialization, effectively untouched. `export/native_hf.merged_state` writes
`input_rows` into the embedding, so the native `input_rows` themselves never moved.

Consequences:
- Nothing in the input marks "a value starts here" or "a value ends here". Both consume markers look like any never-seen
  token.
- The producer document's position 0, `produce_open`, is the same generic token. Every state is computed with that
  token as its only context and its attention sink.
- The head is tied, so the model cannot tell the markers apart as outputs either.

The LoRA+rows core lists marker rows as trainable. Over sft-v6…v9 they moved by about 0.8%. Two explanations remain:
- they are not in the optimizer, or a merge or export drops their updates;
- their gradient is too small, because few consumer calls are trained (§3.2).

This was not checked further (§5). AGENTS.md says Neuralese spans and operators start from text. These markers start
from nothing.

### 3.2 A context-free state that the reader was barely trained to read (strong evidence; the main mechanism)

A typed-text result is written by `write-text`: KV of the text alone, from position 0, after an untrained marker
(`serve/sglang_actor.py:655–681`, `serve/engine.py:1262–1290`). The reader therefore attends to K and V that were never
in its own context:
- The first value token's keys encode "follows the position-0 marker". They do not encode "follows `= `".
- Induction-style copying has no learned entry point into the span. Once inside, it can chain through the span's own
  previous-token information.

This predicts what §2 shows: the gist or a salient name sometimes crosses, and the details fail.

Almost no training covered reading such spans. sft-v9's `neuralese-graph-own` admitted:
- 60 episodes;
- 392 consumer calls;
- 412k consumer target tokens;
- `forced_prefix_calls` 37, the template readouts among them.

These came out of 1,239 Neuralese episodes. The funnel dropped 749 for missing rollouts, 206 for excluded runs and
179 for unseen surfaces (`ops/ei/sft-v9/neuralese-graph-own/build.json`). `neuralese-text` drops every call after an
import.

Expert iteration then selects successes. Most Neuralese successes bypass the read (HELD_FAILURES §3.1 table), so the
reader skill gets no positive data. This is a loop: the reader is untrained because reads fail, and reads fail because
the reader is untrained.

### 3.3 A template readout of stored content is sampled at temperature 1, top-p 0.95 (medium evidence; it compounds the others)

`natlang.read` is a decode readout that should return the stored text exactly. It runs under the episode's sampling law
(rollout `sampling_law`: `temperature 1, top_p 0.95`). Any per-token uncertainty compounds: with per-token accuracy
p, the chance of an exact string is p^n.

This fits the steep length curve in §2: 16% exact below 8 tokens, 0% at 32 or more. It does not explain the flat
first-token choices (`jinja2`), which come from §3.1 and §3.2.

### 3.4 Checked and ruled out (with evidence)

- **The producer adapter.** `natlang_producer_embed` equals `embed_tokens` bitwise for the span tokens and the marker
  rows, so the adapter is the identity. The learner's plain embedding lookup in pass-through is therefore exact. The
  stale producer-table bug of 2026-11-17 is not present in this export.
- **Wrong layer or wrong rows in the capture.** Layer-0 B·x recomputes from the stored embeddings at cosine 0.9997 per
  row. Every layer is present: 12 conv layers with `gated_inputs` (2048 wide) and 12 attention layers with `raw_k`/`v`
  (4 heads × 128). All are finite and of normal magnitude (per-head K norm 12–17).
- **The text or token ids of the state.** `token_ids` decode to the producer's exact text; `positions` run 1..n.
- **A wrong or stale block id at the consumer.** Each reader's `inputs.block`, `native_state` part and import receipt
  name the producer's own block. Equal ids for equal texts are correct content addressing: the bytes are identical.
- **A detached or empty import.** The import is reported `exact-native-state`. `count = len(token_ids)` is greater
  than 0, and a count of 0 is refused at `sglang_actor.py:1496`. Content found only in the state reaches outputs (§2).
- **Prefill CUDA graphs skipping the Python import hook.** `actor.log`: "Disable prefill CUDA graph because some layers
  do not apply Standard GQA". Every prefill reports `cuda graph: False`. `enable_mixed_chunk` is False, and the fork
  raises on mixed or captured batches.
- **The prompt shape.** The tokenizer-decoded aei1 reader prompts match what the code renders. The held
  `provider_exchange` request has the same structure, carrying `template.value = 'decode'`.
- **Weight precision or fork commit.** The failures are the same on BF16 with dbc51ddb72 (Pop) and FP8 with ec27f0809a
  (DGX).
- **Learner and actor splice semantics.** `lfm3_fast_train` (normed K, V and conv B·x via index_put before RoPE;
  separate producer document from position 0) and the fork's `NatlangImport` agree, and so does the native
  `splice_block`.

## 4. Fixes this points to

These are not made here: this task only writes the findings file.

1. **Marker rows.** Find why `input_rows` did not move (optimizer membership, merge, export), then train them. Or
   initialize them from text before training, for example the mean of the embeddings of words such as "value" or
   "quote". This fits AGENTS.md's "start from text".
2. **Write states in the reader's own rendering.** Use the existing `init-body` operation ("text at its rendered
   position in a call", `native-state-client.ts:107`). Alternatively, write the producer document with BOS and a short
   context that matches where the reader will see it, so that the first value token's keys look like in-context text.
   The writer and the learner's pass-through document change together, so learner parity is kept.
3. **Reader data, from any source.** Exact `natlang.read` targets are known: they are the producer's text. Every
   recorded write gives a free supervised reader row, the stored or pass-through state plus its own text as the target.
   There are thousands of these in the aei1 store, independent of task success. A synthetic read-back curriculum is
   another source.
4. **The readout's law.** Decode `natlang.read` greedily, or record it as a deterministic readout. That makes an
   exact copy possible once the reader can read.
5. **The AGENTS.md acceptance test.** A probe that scores a reader on the state, on the same text in context, and on
   a shuffled state, at several lengths.

## 5. Not checked (static scope)

- **Consumer-side numerics in a live engine.** Static work cannot tell whether the fork's import gives the same logits
  as the native `splice_block` on a real request, or whether the radix-cache hit path (a prefix ending inside or after an
  imported span, extra_buffer conv checkpoints at 256-token grids, the SWA pool) matches a cold prefill. These need a
  live A/B: the same reader with the radix cache cold versus warm, and SGLang versus the native engine.
- **How well the base model reads transplanted KV.** This needs the probe in §4.5.
- **Why the marker rows did not move.** The LoRA learner's optimizer parameter list and the factor merge in
  `publish-round` were not read.
- **The llama.cpp and wllama engines.** They served none of these episodes. Their import code (the
  `llama.cpp-neuralese` fork) was not read for this question.
- **Reads with 2–8 states in one reader prompt** (92 of the 710 aei1 reads) were not scored.

Scripts and outputs are in this session's scratchpad (`static2/`): `read_stats.py`, `leak.py`, `block.py`, `layer0.py`,
`markers*.py`. The CPU checks read the export and the block file with `safetensors` slices (≤ 50 MB). The ledger did not
admit the 4 GB unit, because the DGX was full with aei1-r0001-sft.
