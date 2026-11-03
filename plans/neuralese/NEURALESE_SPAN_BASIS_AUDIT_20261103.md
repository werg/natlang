# Neuralese span basis audit (2026-11-03)

## Question

The owner asked where the Natlang/LFM3 system seeds, initializes or supervises a Neuralese span, soft state or
learned prompt-embedding operator from something other than a good token sequence. A good basis is input-specific,
target-relevant content that the consumer actually needs.

The trigger was the learned compaction operators. Their slot outputs were seeded from a fixed cue ("Facts this task
needs from the document:" cycled to K slots) that was identical for every input. At step 0, actual equalled shuffled
in the ablation.

## Method

Read-only.

**Data.**
- Edge-level sampling through ledger units `nlseed-edges1` and `nlseed-rl2` (low priority, 3–4 GB):
  - up to 200 rows per source, reservoir-sampled from the first 20,000 rows;
  - 3,942 import edges in total.
- For each edge the sample records:
  - the decoded payload;
  - the consumer context immediately before the import;
  - the consumer's supervised target and its lexical overlap with the payload;
  - whether the payload's first 60 characters reappear in the first 300 characters of the target ("verbatim").
- The 15,703-row census from the recipe review (`RECIPE_REVIEW_20261103/NEURALESE_CAPABILITY_COVERAGE.md`) was reused
  for row masses.
- Scripts and raw edges: `/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/seedaudit/`.

**Code.** The reading covered:
- converters in `scripts/neuralese_data/` and `ts-host/scripts/inline-curriculum/`;
- `ts-host/src/compiler/neuralese-conversion.ts`;
- `T/data`, `T/model`, `T/tune`, `T/rl` and `ts-host/src/neuralese`.

`T` = `training/neuralese/natlang_neuralese`.

**Sources.**
- Production: `semantic-ticket-quality-successor-1102/train-sources.json` (7b99d86d…, 401,623 rows).
- Corrected mix, not yet trained: `ops/corrected-mix-diverse-1103/train-sources.json`. It withdraws 305,282 graphs,
  108,040 of them under `template-reader-cap`, and appends 5,455 rest27 chain graphs.
- RL: `full-core-effects-20261102` (61,399 cases) and `natlang-rl-composed-neuralese-20261103` (7,889 cases).

## Findings, sorted by severity

Severity combines the training mass affected with how bad the basis is.

| # | Sev | Where | Basis today | Why it is lazy | Good basis | Fix (effort; S = Sonnet-sized) |
|---|---|---|---|---|---|---|
| 1 | **H** | JOLT+/privileged context for every Neuralese **producer child** in RL: all 15,204 `Neuralese<string>` reader cases (40 families) and all 7,889 composed relay/typed/fan-in/iterate/deep cases | **Nothing.** 0 of 23,093 cases have `curriculum.reference.children`, `decisive` or `evidence` (`rlcheck.json`). By `T/rl/privileged_context.py:385-398` (`applicable`), a child sees only its own `child_value` facts, so producer children get no privileged context. Only the root sees the final `expected_result`. | Producers (`gather`, `refine`, `read_page`, `find_rows`, `profile`, `open_notes`/`absorb`, `explore_program`, `read_file`…) learn their notes only from sparse final-outcome reward. No teacher signal says what a good note contains, and no SD target exists for a note. The upstream gold intermediates were loaded and then discarded. `scripts/rl_tasks/sources/composed.py:159-164` builds MuSiQue `bridges` (per-hop answers) and uses them **only as wrong-answer checks** (`_doc_case(..., item['bridges'])`, `:605-607`). | Per producer helper, a gold or checked note in the child's declared type: MuSiQue decomposition answers plus supporting paragraph sentences; HotpotQA `supporting_facts` sentences; 2Wiki evidence triples; HybridQA/TabFact gold cells and rows; QuALITY answer-bearing sentences; exact caller/field lists for the codebase families; for `profile`, the gold facts as `{subject, relevant, facts}`. Where no gold note exists, a deterministic mechanical one (`T/data/operator_seeds.py` `answer_lines`/`overlap_sentences` over the child's own inputs). | Emit `curriculum.reference.children = [{match, value}]` per producer helper. `child_value` facts then reach the guidance writer and the text-twin through the existing visibility rule (`answer_fits` must accept `Neuralese<string>`). The same values become SD/SFT targets for successful rollouts. Do composed.py first: its loaders already hold the data (S, ~1 day). Then the `rl_tasks/sources/*` reader families, one per source (S each). Leave held/dev cases unchanged. |
| 2 | **H** (mass) / M once the corrected mix is live | Inline-instruction "communication" edges. The inline `nl` template literal is converted into a `Neuralese<string>` producer→consumer edge (`ts-host/src/compiler/neuralese-conversion.ts:1490-1514`, `source_kind: 'inline-instruction'`). | **Production source:** 117,789 big2 rows. In the sample, 90% (`big2-qualified-unique`) and 71% (`offload-9436`) of edges carry the identical 23-token payload "Do what item.request asks, using only item.evidence. Follow item.answer_format exactly and return only the answer." (template at `ts-host/scripts/inline-curriculum/build-port-record-calls.mjs:142`). Same pattern elsewhere: the loghub v9 urgency rubric from a `.nl` file (`inline.mjs:219`) is 20 of 56 v9 edges, and the semantic-iterate seed note "the supplied draft is only a placeholder; no evidence has been reviewed" (`semantic-iterate-worlds-v15-soft-builder.mjs:20,85-89`) is 14% of the sampled t1-communication payloads. | As a *soft instruction*, the real instruction text in the real call is a sound basis. As *communication* data it is constant: one payload over ~99k edges, a consumer import labelled `Instructions:`, and a consumer target (median ~6.7k tokens) that mostly reads `item.evidence` regardless. It teaches no producer to write task-specific state, yet it carried 7× communication quotas, about 76% of all required visits. | Communication edges need value payloads: child answers, notes, extracted facts. Instruction spans should be accounted as soft instructions: one per distinct body, not per reader. | **Partly done in the corrected mix** (untrained): 108,040 template readers withdrawn, remaining template rows quota-capped at 1–2 visits (`classification/template-quota-classes.json`), family `native-communication-template`. **Remaining:** (a) make the converter record inline-instruction edges with an explicit `instruction` kind, so family/quota/focus accounting never counts them as communication (S); (b) cap readers per identical payload in the generic quota code instead of a one-off classification (S); (c) launch the corrected mix (supervisor). |
| 3 | **H**, now fixed in the working tree; residual **M** | Learned compaction operators (`T/model/learned_operators.py`) | **Previously:** K slot rows copied from a constant per-operator `slot_text`, cycled to K, for all four operators: compact-value "Exact names, numbers, paths and structure…" ×64, compact-params ×32, compact-history ×128, extract ×32. **Now** (uncommitted working tree, compaction-operator agent): a variable-length *seed* producer span from `data/operator_seeds.py`, CE-supervised with an annealed schedule (`seed_schedule`, `learned_operators.py:326-365`). It applies to every operator through the one renderer (`data/operator_graph.py:118-158`) and `operator_cases_common.offer` → `seed_for`. Serving generates up to `seed_cap` (`serve/engine.py:918`). Prompt rows stay text-initialized from the real prompt (fine). | The redesign covers all four operators. Residual gaps, all below gold quality: <br>• Gold/checked seeds are supplied only by `broad_extract` and `web_search_r1`. `compact_history`, `compact_params` and `tool_digest` fall back to mechanical seeds. <br>• `history_outline` (`operator_seeds.py:106-129`) is input-specific but not target-related: last 20 actions, paths, errors and the last step, with no ranking by what the continuation needs. <br>• `compact-value` skips `overlap_sentences` and falls to identifiers plus a positional head/tail `outline` (`:145-150`). The head/tail fallback is generic. | compact-history: the bgkit `compaction_v2` summary, or a checked account of the identifiers/paths/errors that the target continuation actually references. compact-value/params: question/target-overlap sentences before the outline. tool_digest: its exact code-derived refs as `checked`. | (a) Rank history items and value sentences by overlap with the consumer target and question (S). (b) Supply `checked` seeds from tool_digest refs and compaction_v2 where present (S). (c) Report ablation actual vs shuffled by seed provenance (already planned, `neuralese_ablation.span_report`). Owner: compaction-operator agent (FALLOW C1/C3/C8). |
| 4 | M | Static long-context graphify: `T/data/static_native_context.py:14-52` (`state_tokens=16`). Default path for over-position acquired rows: `transform_acquired_ordinary_context.py:203`, `graphify_pending_acquired_context.py:57`, `build_acquired_full_session_graphs.py:492`, `prepare_*static*`. | Each ≤64k-token chunk exports **its last 16 tokens**, wherever the chunk boundary falls, with zero producer weight. Samples: `' The project to the final: The final the final…'`, `') allowedMethods(path string) []string {\n+\tmethods := make([]'`, consumer context `'GUI'`, `'requirements'`. | The token basis is an arbitrary mid-text fragment, not a summary. Whatever the consumer gets beyond those 16 tokens' own KV comes only through the hybrid state. Mass is small today (11 + 8 rows in production; withdrawn in the corrected mix), but this is the default for every future long-context conversion. | A target-related summary of each chunk as the producer span: the `compact-history` mechanical/gold seed of that chunk, ranked by what the consumer target needs. Better still, route these rows through the learned `compact-history` operator. | Replace the tail span with an appended seed span (`operator_seeds.history_outline`/`overlap_sentences`) under producer CE, or emit `compact-history` operator cases instead (M). |
| 5 | M | rest27 D7 chains (5,455 rows, corrected mix) and `t1-communication-predecessor` (4,911 rows). Renderer `scripts/neuralese_data/native_chain_render.py:26-60`. | Real returned child values: rest27 has 544 distinct payloads in 847 sampled edges (median 81 tokens); predecessor has 196 distinct in 363 (median 35). | The basis is good, but the consumer task is often a copy. In ≥29% of rest27 edges (static-lambda-v2 digests, s7k translation maps) the target re-emits the payload verbatim. In predecessor rows the parent re-emits an answer map containing the payload (99% lexical overlap). This teaches lossless relay, not compaction or use. | Keep these as relay/readback supply. Weight towards chains where the consumer transforms the payload (bgkit-knowledge, port-record-s3, v13-r2). | Flag verbatim-relay edges in the census and cap their quota or CE weight (S). |
| 6 | M–L | loghub/t1-communication/qwen15 label edges (~780 rows) | Gold labels and JSON verdicts: `international_fees`, `{"sufficient":false,"finding":null}`. Median 4–12 tokens, input-specific. | Correct but low-information. In 68% of loghub `t1-partial` edges the consumer target shares no word with the payload. Recurring constant payloads: `{"sufficient":false,"finding":null}` is 7% of qwen15 edges. | Fine as label routing. Richer notes need the source's evidence (see #1). | None beyond #1 and #2 (S). |
| 7 | L | Teacher-text encoders for Luna typed episodes (`T/data/provider_native_graph.py:82-86`, `provider_static_graph.py:94-97`; ~10 episodes) | Exact authenticated typed body (`Neuralese<FaultDiagnosis>`, `ComponentSelection`…). | The content is good, but it is written **context-free** (produce-open, body, produce-close; no type/purpose/call context), with zero target weight. One sampled memo is read 22 times. | The same body written in its real call context. | `native_init_body` with the replay messages already loaded at `provider_native_graph.py:70` (S). |
| 8 | L | Hindsight hints (`T/rl/hindsight.py:113-119`) and memetic program guidance (`T/rl/memetic_program.py:75-91, 765-776`); neither is live | Writer-produced hints from verified executions; teacher-authored guidance from failure digests. | Good, specific content, but it is re-encoded bare via `/write-text` rather than in the executor's call. | The same text in context. | Use `native_init_body` in a real executor call (S–M). |
| 9 | L | Marker rows (`T/model/shared_state.py:1186, 1207`, `provision_role_tokens`) | All four new marker rows start as the mean embedding and mean head row, so they are identical. | Produce and consume are indistinguishable at step 0. This affects only new lineages; existing checkpoints restore trained rows. | Structurally analogous existing tokens (tool-call open/close), or mean plus distinct offsets. | S. |
| 10 | L | Standard-library combinator bodies (`ts-host/src/neuralese/combinators.ts:38-66, 99-125`); runtime API, not training | A generic description per combinator, written after one fixed toy input ("The meeting moved to Tuesday…", `{city:'Lyon'}`). | The context is a constant toy call. | Real recorded combinator calls, one per dialect/type. | M (needs a call corpus). |
| 11 | L | `T/tune/learned/writer.py:61,138` (`extra_slots` padded with `','`); bank components as random Gaussians (`meta_bank.py:305-318`); growth keys written bare (`growth.py:55`) | Pad token; random; bare description. | Latent: every caller passes `extra_slots=0`. The random bank is a deliberate comparison baseline. Keys never appear in the decision context. | Target continuation text; fitted components (`growth.source_component`); keys written in context. | S each, when used. |

### Places that are fine (checked)

- **Soft instructions and skills in the tuning lane** (`T/tune/soft.py:76-113` → `engine.native_init_body`,
  `serve/engine.py:1150-1245`; `ts-host/src/neuralese/text-init.ts:52-75`).
  - The binding's real instruction text, or the disclosed skill document, is written at its real position in the
    first real training turn of that binding, with that turn's tools.
  - There is no random, zero, mean or pad fallback: a text without a unique position is skipped.
  - Minor: one trajectory (the first in sort order) stands in for all of them.
- **Guidance writer / JOLT+ facts** (`applications/guidance-writer/writeGuidance.nl`,
  `ts-host/src/native/training-guidance.ts:136-191`, `T/rl/privileged_context.py:328-383`).
  - The facts are per-world and custody-checked: expected result, files, effects, reference solution, child values,
    attempt feedback.
  - The writer is the policy itself, writing in its real call.
  - The text-twin fallback passes the selected fact texts verbatim.
  - The weakness is upstream: the cases supply no child facts (#1).
- **Typed `Neuralese<T>` in SFT** (`native_chain_render.py:26-60`,
  `build_native_semantic_pairs_e2e.py:801-835, 996-1023`).
  - The producer span is the exact typed value inside the real `return_result` call.
  - The whole call is the CE target.
- **Reasoning-boundary override rows** are exact copies of the `<think>`/`</think>` rows.
- **The rank-8 `LowRankTokenEmbedding`** (`shared_state.py:26-38`) starts as an identity residual (Kaiming `down`,
  zero `up`).
- **Native adapters, coefficient policy, import channel and decision readout.**
  - Adapters: LoRA with B=0, and zero coefficients on a weight-derived basis.
  - Import channel: noise, by design.
  - Decision readout: a scalar loss weight, with no learned rows.
- **Composed RL `smoke_children`** ("row note", "first note", `SMOKE_PROFILE`) are used only by
  `structure_smoke.mjs`. They are not training targets.
- **Semantic-iterate per-pass notes** are mechanical summaries of the gold pass states (`noteFor`). Only the initial
  note is the constant placeholder, which the program asks for by design. It is folded into #2.

## Priorities

1. **#1:** child reference values for Neuralese producers in RL. This is the only finding that affects the live Neuralese
   supply the recipe review ranked highest. Without it, JOLT+ adds nothing to producers, and no successful note
   exists to self-distil from except the policy's own.
2. **#2:** launch the corrected mix, and make inline-instruction edges structurally non-communication.
3. **#3:** finish the operator seed residuals (target-ranked history/value seeds, checked tool_digest/compaction_v2
   seeds) before the first operator training window. Then ablate by seed provenance.
4. **#4:** replace the 16-token tail with a seeded summary span before any further long-context conversion.

Status 2026-11-03 (S2 and #4 implemented for future conversions and quota computation; the corrected-mix package's own
withdrawals are unchanged):

- **S2.** Inline-instruction writer/read pairs carry `edge_kind: 'instruction'` (converter occurrences and mined edges).
  Chain assembly writes `instruction: true` on those graph imports; `communication_classification` then does not count
  them (a graph with only instruction imports is ordinary: no communication family, quota or exposure), and the
  per-edge pair builder skips them. `communication_minimum_quotas` caps the repeated quota at
  `DEFAULT_MAX_READERS_PER_PAYLOAD = 3` readers per identical payload (token-slice digest `payload_sha256` in new index
  entries; rows from older indexes have no digest and keep the quota, reported as `unknown_payload_rows`), recorded in
  `communication_quota_policy.payload_reader_cap`.
- **#4.** `graphify(..., tokenizer=...)` exports a seeded chunk summary (`operator_seeds.chunk_summary_seed`:
  question-selected sentences, exact identifiers, outline) as each chunk's producer span, under bounded producer CE
  (schema `/2`, seed provenance recorded); without a tokenizer the legacy tail export (`/1`) still replays and verifies the
  earlier rows. `build_acquired_full_session_graphs` passes its tokenizer.

Items marked S are suitable for a Sonnet helper given this table.
