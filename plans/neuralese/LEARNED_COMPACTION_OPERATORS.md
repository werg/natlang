# Learned Neuralese compaction operators (2026-11-03)

Owner intent: AGENTS.md "Neuralese compaction is the intended runtime design" and the text-initialized learned
operators follow-up. This note records the design decisions; the converter work is specified in
`COMPACTION_OPERATOR_CONVERTERS.md`, the open items are rows in `FALLOW_REGISTER.md` (C1-C8).

## Natlang-native decomposition

| Part | Form | Why |
|---|---|---|
| When to compact (value over a size threshold, conversation near its limit, a program asking for an extract) | crisp runtime policy with a pluggable NL implementation (`ts-host/src/runtime/pluggable.ts`); crisp default | a size threshold is exact; deciding *what a value is for* is the operator's job, not the trigger's |
| What to keep | the learned operator (prompt rows + its generated producer span), conditioned on the receiving purpose in text | this is the semantic judgment; it is what training changes |
| Writing the state | prefill of the operator site, greedy generation of the producer span up to its cap, capture of the span | mechanism; exact |
| Seeding the span in training | gold/checked spans from the converter, else one shared deterministic mechanical extractor (`data/operator_seeds.py`) | exact, auditable; never invented summaries |
| Showing it | the existing consume-marked native state part plus a one-line text note saying where the full value is | the full value stays reachable (`read_page`, eval) |
| Is it used | the shared ablation (`eval/neuralese_ablation.py`) | an exact measurement, not a policy |

## Operators (model/learned_operators.py)

| Operator | Span cap (tokens) | Placement | Runtime site | Dedicated data target |
|---|---|---|---|---|
| `compact-value` | 512 | system prompt | large tool/service outputs, documents, collections in `renderValue` | 25k |
| `compact-params` | 384 | system prompt | a call's large arguments in its opening listing (`inputsListing`) | 25k |
| `compact-history` | 1024 | final user message after the conversation | `compact_history` and the automatic near-limit compaction | 25k |
| `extract` | 256 per document | system prompt; one site per document (fan-in) | Neuralese-prompted extractor feeding a parent lambda | 30k |

Registry version 2 (the correction below); version-1 slot bindings are not loadable and were never trained.

## Owner design correction (2026-11-03, binding)

Version 1 gave each operator K fixed **slot rows** seeded from a generic slot text, written by one prefill. The
smoke ablations showed the expected consequence: an untrained slot state carries no input-specific content (text-init
compact-params actual 4.375 vs shuffled 4.370; history 0.556 vs 0.563), while the existing written-text producer path
is strongly used (rest27: 0.415 vs 1.32). The owner corrected the design:

1. The operator's output is a **generated producer span** (variable length up to a per-operator cap) seeded as a
   **text summary related to the consumer's target**, with CE supervision on the summary, so the imported state
   carries input-specific content from step 0. This is the existing producer->consumer path.
2. Seeds per operator, never fabricated: extract uses Hotpot/2Wiki supporting facts, MuSiQue supporting paragraphs,
   Multi-News reference summaries (greedy ROUGE oracle sentences), the SQuAD answer sentence, QMSum/FeTaQA relevant
   spans; compact-history uses actual compaction notes where they exist plus SWE continuation context; compact-value
   uses code-derived recall targets for digests and web_search answer-bearing spans; compact-params uses
   question-relevant or answer-bearing spans.
3. Where no gold or checked summary exists, a **mechanical** seed is used rather than dropping the source: answer
   lines (sentences containing the answer string), overlap sentences (lexical overlap with the question/target),
   identifiers (exact paths, names, numbers) with a structure outline, an outline with head/tail (JSON keys, line
   counts, headings), or for conversations a history outline (actions, paths, errors, last step); budget scaled to
   input size. One deterministic helper (`data/operator_seeds.py`) serves every converter and the renderer. Every
   seed records its provenance `gold` | `checked` | `mechanical:<method>`, so learning can be compared by seed type.
   Finding: the bgkit `compaction`/`compaction_v2` parquet targets are the *next assistant message*, not compaction
   notes, so compact-history has no gold notes there and uses `mechanical:history-outline`.
4. Then **morph**: the whole graph trains end to end; consumer-outcome gradients flow through the producer span's
   state while the seed CE weight anneals on a declared schedule (`operator_seed_schedule {start, end, updates}`,
   default 1.0 -> 0.1 over 2,000 updates from its adoption; the floor keeps the serving-time generator anchored to
   readable, input-specific content). The text-initialized prompt rows stay.
5. Optional later: a single-prefill slot writer distilled from the trained spans (state matching). Not the start.
6. Measurement: the ablation (actual / shuffled / zeroed / text-cut) plus span length vs input size and use deltas by
   seed provenance (`neuralese_ablation.span_report`).

## Parameterization

Each operator owns **prompt rows** (one learned embedding row per token of its prompt text, at the prompt's actual
position in the rendered site) and writes a **producer span** after `produce_open`.

- *Text initialization in the real call.* Prompt rows start as exact copies of the pinned embedding rows of the
  prompt's own tokens, and the rows sit where the text sits in the real rendered site, so at step 0 the site
  computes exactly what the text-prompted site computes (`test_text_initialized_rows_reproduce_the_text_forward_exactly`).
- *Why embedding rows, not stored KV soft states.* Soft states (`tune/soft.py`) are KV/hybrid spans computed by one
  checkpoint. In production the backbone trains jointly, so a stored KV parameter goes stale against the moving
  layers. Embedding rows flow through the current backbone on every pass, and the identical arithmetic serves
  training, ablation and inference. For a frozen deployed checkpoint the prompt part can additionally be folded into a
  cached prefix state (exact KV prefix reuse), which is a performance cache, not a different operator.
- *Generation cost.* Serving generates the span greedily (deterministic given the site), bounded by the cap and
  ended by `produce_close`. That is a decode per large value; the runtime trigger stays crisp and size-gated, and the
  distilled single-prefill writer (5. above) is the later cost reduction once trained spans exist.
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
`{operator, version, node, kind: prompt|seed, start, stop[, provenance]}` (node-local token positions).
`resolve_spans` checks prompt token IDs against the operator's bound IDs, and that a seed lies inside a producing span
within the operator's cap. Seed tokens and the closing `produce_close` carry CE weight 1 in the row;
`learned_operators.seed_weighted` scales them by the schedule's current factor per update. A trainer without
operator rows sees the operator's text tokens, i.e. its text-initialized form, and the seed at full weight.
`operator_conversion` records each seed's provenance, token count and clipping, the span and producer-site token
counts. Exact producer/consumer
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
   The seed CE anneals per `operator_seed_schedule` (recorded with its adoption step; a changed schedule is a new
   adoption).
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

These are the starting point the trained operators must beat; they are smoke-sized, not held-world results. The
three operator smokes used the retired version-1 slot design; seeded-span graphs are re-rendered and re-measured.
