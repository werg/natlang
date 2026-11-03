# Learned Neuralese compaction operators (2026-11-03)

Owner intent: AGENTS.md "Neuralese compaction is the intended runtime design" and the text-initialized learned
operators follow-up. This note records the design decisions; the converter work is specified in
`COMPACTION_OPERATOR_CONVERTERS.md`, the open items are rows in `FALLOW_REGISTER.md` (C1-C8).

## Natlang-native decomposition

| Part | Form | Why |
|---|---|---|
| When to compact (value over a size threshold, conversation near its limit, a program asking for an extract) | crisp runtime policy with a pluggable NL implementation (`ts-host/src/runtime/pluggable.ts`); crisp default | a size threshold is exact; deciding *what a value is for* is the operator's job, not the trigger's |
| What to keep | the learned operator (prompt rows + slot rows), conditioned on the receiving purpose in text | this is the semantic judgment; it is what training changes |
| Writing the state | one prefill of the operator site, capture of the K slot positions | mechanism; exact and cheap |
| Showing it | the existing consume-marked native state part plus a one-line text note saying where the full value is | the full value stays reachable (`read_page`, eval) |
| Is it used | the shared ablation (`eval/neuralese_ablation.py`) | an exact measurement, not a policy |

## Operators (model/learned_operators.py)

| Operator | Slots K | Placement | Runtime site | Dedicated data target |
|---|---|---|---|---|
| `compact-value` | 64 | system prompt | large tool/service outputs, documents, collections in `renderValue` | 25k |
| `compact-params` | 32 | system prompt | a call's large arguments in its opening listing (`inputsListing`) | 25k |
| `compact-history` | 128 | final user message after the conversation | `compact_history` and the automatic near-limit compaction | 25k |
| `extract` | 32 per document | system prompt; one site per document (fan-in) | Neuralese-prompted extractor feeding a parent lambda | 30k |

## Parameterization decision

Each operator owns **prompt rows** (one learned embedding row per token of its prompt text, at the prompt's actual
position in the rendered site) and **slot rows** (K learned embedding rows after the input, inside a producing span).
The site's output is the native state of the K slot positions.

- *Text initialization in the real call.* Prompt rows start as exact copies of the pinned embedding rows of the
  prompt's own tokens, and the rows sit where the text sits in the real rendered site, so at step 0 the site
  computes exactly what the text-prompted site computes (`test_text_initialized_rows_reproduce_the_text_forward_exactly`).
  Slot rows start as the embeddings of the operator's slot text, cycled to K.
- *Why embedding rows, not stored KV soft states.* Soft states (`tune/soft.py`) are KV/hybrid spans computed by one
  checkpoint. In production the backbone trains jointly, so a stored KV parameter goes stale against the moving
  layers. Embedding rows flow through the current backbone on every pass, and the identical arithmetic serves
  training, ablation and inference. For a frozen deployed checkpoint the prompt part can additionally be folded into a
  cached prefix state (exact KV prefix reuse), which is a performance cache, not a different operator.
- *Why fixed slots, not a sampled text summary.* The existing `view` Neuralese instance samples a text summary per
  value: a full generation for every large value. Slots cost one prefill, give a true fixed-size bottleneck, and
  their content is learned only through consumers that import them (the model's own outputs are the only targets;
  slots carry zero loss weight).
- *Bank mixtures and adapter components.* Not the first parameterization: a per-operator LoRA component changes the
  whole site's computation and needs the bank's per-program coefficient machinery, which the bank meta-learning agent
  is building. Promotion path: once an operator is measured useful, the bank stage may add an operator-specific
  component (coefficient one-hot on operator sites), trained on the same operator data. Coordinated through the
  fallow register row C7.

Rows are FP32 masters on the AdamW group of learned rows (same routing as marker rows and the embedding adapter),
optimized jointly with the backbone in the production stage. The producing positions also receive the existing
`LowRankTokenEmbedding` adapter, as every producer does.

## Graph format

Operator data are ordinary `natlang.neuralese-native-graph/1` rows with one extra top-level list, `operators`:
`{operator, version, node, kind: prompt|slots, start, stop}` (node-local token positions). `resolve_spans` checks the
token IDs at each span against the operator's bound token IDs and that slots lie inside a producing span. A trainer
without operator rows sees the operator's text tokens, i.e. its text-initialized form. Exact producer/consumer
rendering is specified in `COMPACTION_OPERATOR_CONVERTERS.md` §1.

## Use measurement and promotion

`python -m natlang_neuralese.eval.neuralese_ablation` scores consumer targets with the actual imported state, with
state shuffled across rows (a derangement) and with zeroed state, plus optionally the runtime's text cut of the
same value. Promotion of an operator to runtime default requires, on held worlds: NLL(actual) below shuffled, zeroed
and text-cut, and a matching paired task-outcome check in the live runtime.

## Pipeline (2026-11-03)

1. Converters (Sonnet, `COMPACTION_OPERATOR_CONVERTERS.md` §2-§6) write cases to
   `/mnt/external/natlang-compaction-operators-20261103/cases/<operator>/`.
2. `scripts/neuralese_data/build_operator_graphs.py --cases <file or cases root>` renders graphs plus text-cut
   siblings into `.../graphs/<operator>/<source>/` under the current protection projection (a ledgered watcher
   renders each case file once its manifest lands).
3. `scripts/neuralese_data/operator_stage_sources.py --out DIR` writes the operator train/held source manifests and
   per-operator counts against the data targets.
4. Training: a `native_state_sft` stage with `learned_operators: all` appends the operator train sources to the joint
   SFT continuation through its recorded source transition. Rows attach text-initialized (recorded
   `learned_operator_history`, new zero Adam moments), train at the learned-row rate and persist in `interface_state`.
5. Measurement: `python -m natlang_neuralese.eval.neuralese_ablation --checkpoint C --graphs held --text-cut
   held-textcut [--text-init-operators]` before and after training; promotion as above, then the runtime default
   (`compaction: nl`) after the live check.

## Measurements

Baselines on full67 (`ops/bf95-quality-source-zero-sgd-v2-1102`), weighted consumer-target NLL; reports under
`/mnt/external/natlang-compaction-operators-20261103/ablation/`.

| Graphs | Rows | actual | shuffled | zeroed | text-cut | Reading |
|---|---|---|---|---|---|---|
| rest27 bgkit-knowledge graphs (not yet in production; text-prompted verbatim producers; 41 longer graphs skipped) | 32 | 0.415 | 1.323 | 1.317 | — | consumers strongly use imported **written** text state (+0.91 nats/token without it; actual better on 78%/75% of rows) |
| compact-params smoke (S1 repo_qa, text-init rows) | 21 | 4.375 | 4.370 | 4.614 | 4.132 | untrained state carries no row-specific information; the text cut is better |
| compact-history smoke (S1 SWE checked, text-init) | 10 | 0.556 | 0.563 | 0.568 | 0.568 | small row-specific use |
| extract smoke (hotpot, text-init) | 12 | 7.742 | 7.802 | 8.458 | 6.186 | documents are short, so the "text cut" is the whole text (upper bound) |

These are the starting point the trained operators must beat; they are smoke-sized, not held-world results.
