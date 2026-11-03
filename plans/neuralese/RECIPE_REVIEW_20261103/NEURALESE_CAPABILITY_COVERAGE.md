# Does the current LFM3 recipe instill the Neuralese and Natlang behaviours we want? (audit, 2026-11-03)

## Owner question

> "Are we going to instill the desired neuralese with deep call graphs, use of skills, general coding, directory
> reducers, iterateOn, decisions? Do we have a wide range of directly learned neuralese prompt embedding operators
> and training data for them?"

## Short answer

No, not with the current recipe and data.

1. **Neuralese passes through one template.** 30.7% of training rows (123,488) are producer-to-consumer graphs. About
   99% of them come from one scripted port-record template. In ~84% of those, the exported state is the same
   23-token instruction string. The remaining Neuralese data is effectively only:
   - 473 loghub graphs;
   - 191 Qwen graphs;
   - about 10 Luna `Neuralese<T>` episodes.

   Those are the only rows with real depth, fan-in or fan-out. Producer-consumer depth of 2 or more appears in
   0.05% of rows.
2. **The natlang constructs are mostly the same template.** `iterateOn`, directory reducers and per-item `nl`
   appear in 37-38% of rows, but almost all of those are the same `answer_collection()` program. Excluding it:

   | construct | share of all rows |
   |---|---|
   | `iterateOn` | 1.3% |
   | directory reducer | 3.4% |
   | awaited `nl` | 2.8% |
   | `decide()` calls (some malformed) | 0.22% |
   | skills bound in context | 0.08% |
   | stored guidance or external state | 0 |
   | decision-readout graphs | 0 |

3. **Deep call graphs are rare and shallow in kind.** Invocations at depth ≥4 are about 0.7% of rows, and nearly
   all of them are leaf boolean classifiers.
4. **"Coding" is mostly the same template.** The `coding` family is 96% the port-record template answering
   code-continuation requests as strings. Tool-using coding is about 2k rows plus about 60k acquired multi-agent
   SWE sessions, and most of those sessions supervise only the final report.
5. **The learned operator range is narrow.** The trained model has exactly two Neuralese-specific learned operators:
   - 4 marker rows;
   - a rank-8 embedding adapter on producing positions.

   It also has the reasoning-token override rows. Decision readout, stored guidance or soft states, LoRA
   bank/mixtures, the import channel and the metalearner exist in code, but none is trained in the current stage
   and none has data in the current source.
6. **RL is not running.** This lineage has 0 trained RL rounds and `mandatory_sft_only: true`. Its curriculum
   (61,399 cases) does contain 15,204 `Neuralese<string>` reader tasks with fan-in, which are the most promising
   existing Neuralese supply.
7. **What has actually been earned is mostly the template.** 1,044 graph visits of the 1,142,578 required. 76% of
   the required visits are 7× quotas on the template-dominated communication rows. The pending focus promotes 968
   template communication rows and 1,010 template iteration rows.

The details and evidence follow.

## Method and evidence

- **Source:** `/home/werg/data/lfm3-rl-dgx-20261029/ops/semantic-ticket-quality-successor-1102/train-sources.json`
  (sha `7b99d86d…`). It has 182 leaves and 401,623 rows. The per-source row counts reconcile exactly: 135,944
  fully streamed, plus receipts for big2 (108,353), unchanged-leaf (45,781), novel-position (13,802),
  t5-port-coding (71,141) and t2-port-research (26,602).
- **Recipe:** `earned179-joint-r38-a4-window64-1103/recipe-runtime.json`, one stage `native_joint_sft_v4`, with the CE
  policy `semantic-ticket-quality-successor-1102/ce-policy.json`.
- **Sampled decode census** (`census.py`):
  - Up to 160 rows per leaf by reservoir sampling, or 25 per D32 leaf; 15,703 rows over all 182 leaves.
  - Decoded with the pinned LFM3 tokenizer.
  - Neuralese structure is read from `imports`, `producing`, `external_states` and `decisions`.
  - Construct flags are regexes over assistant spans (`<|im_start|>assistant … <|im_end|>`) and over each node's
    opening.
  - Population estimates weight each leaf by its exact row count.
- **Teacher-decision join** (`idiom.py`): 127,198 of the 127,211 train rows that map to `teacher-program:…:decision:N`
  were joined to `runs/lfm3-mixture/scores/idiom-717f2f18e004`. This gives invocation and parent call depth,
  curriculum family and signature.
- **RL cases:** `/home/werg/data/broad-rl-tasks-20261031/full-core-effects-20261102-…/cases.jsonl` (61,399 cases),
  structure only.
- **Progress:** `earned84-joint-r38-a4-window64-1103/state.json`.

Scripts, raw per-row features (`census.jsonl`) and logs are in
`/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/audit-nl/`. Everything ran
read-only through memory-ledger `audit-nl-*` units at low priority.

Caveats:
- The five receipt-counted large leaves were sampled from their first 3,000 rows only.
- Regex construct detection is approximate. Spot checks showed that some `decide`/`folder.apply` hits are prose in
  `<think>` text, so true code use is lower than reported.
- The idiom-score depth uses `parent_invocation` where recorded (98k of 127k rows); a missing parent counts as
  depth 1, so depth is a lower bound.

## 1. Neuralese graphs: how many, what shape, how deep

The native-communication family has 123,488 graph rows (30.7% of rows; the family census says 123,477).

| source | rows | shape | what is exported |
|---|---|---|---|
| big2 port-record (`big2-qualified-unique` + offload) | 117,789 | 2 nodes, `answer_collection()` → `nl@eval:8(item)` | 84% of sampled edges: the identical 23-token inline-template literal "Do what item.request asks, using only item.evidence. …". 16%: a child's answer read back by the parent as an answer map. |
| t1-communication-predecessor | 4,911 | same template, child → parent | the child's answer; the parent target re-emits the answer map containing it as text. 24 rows are the instruction literal. |
| loghub graph successor | 473 | 4-20+ nodes; DAG depth 2-5 in 30%; fan-in ≥2 in 43%; fan-out ≥2 in 69% | mostly short labels and criteria (median 8 tokens, e.g. `silver maple`, `international_fees`, an urgency rubric) |
| qwen15 complete graphs | 191 | 3-20 nodes; fan-in ≥2 in 40% | short typed JSON verdicts (`{"sufficient":true,"finding":"fraud"}`) |
| t1-communication cohorts | 115 | 2 nodes | labels (median 7 tokens: `talk.politics.mideast`, `MeanOfTransportation`). One recurring payload is "This is only a placeholder; no evidence has been reviewed." |
| Luna native episodes (telecom, relational, sales, coedit, deep-codebase, tatqa) | ~10 | 7-35 nodes, `Neuralese<FaultDiagnosis>` and similar typed results, fan-out 2-6 consumers | real typed results such as diagnoses, briefs and component selections, written by `teacher-text-encoder` nodes |
| acquired static long-context graph pairs | 11 | 16-token tail spans of 64k-token chunks imported by one consumer | context compression of agent traces |

Payload diversity in the sample:

| group | payloads | distinct | top payload share | median tokens |
|---|---|---|---|---|
| big2 | 640 | 97 | 84% | 24 |
| predecessor | 160 | 158 | 1% | 50 |
| loghub | 508 | 182 | 6% | 8 |
| t1-communication | 115 | 44 | 14% | 7 |

The CE policy already recognizes the big2 template: its `decision_factors` scale the byte-identical producer target
(`6562e294…`, 93,229 occurrences) by 0.01. The edges themselves and their 7-visit quotas remain.

Graph depth and fan-in across all rows (weighted):

| property | rows | share |
|---|---|---|
| producer-consumer DAG depth ≥2 | ~170 | 0.05% |
| fan-in ≥2 | ~290 | 0.07% |
| fan-out ≥2 | ~400 | 0.10% |
| one span imported at ≥2 positions | ~19k | 4.7%, almost all template multi-read |

Natlang call depth comes from the teacher-decision join:
- Of the 127,198 teacher decision rows, 104,976 (82.5%) are the `port_record_calls` curriculum (depth 2).
- Across all of them:

  | invocation depth | rows |
  |---|---|
  | 1 | 29,164 |
  | 2 | 92,110 |
  | 3 | 3,131 |
  | 4 | 1,564 |
  | 5 | 833 |
  | 6 | 354 |
  | 7+ | 42 |

- Rows at depth ≥4: 2,793, which is 0.7% of all training rows. Depth ≥3: 5,924 (1.5%).
- Trajectories with max depth ≥4: 1,032 of 27,911.
- The deep rows are dominated by leaf classifiers inside reducer → per-item → criterion chains, for example
  `nl(input: string): boolean` (1,268) and `nl(observation: string): Finding` (486). By family: workbench_tickets
  1,819, recurrence_inline_graphs 1,541, folder_criteria_reducer 1,097, dataset_workbench 770.
- The Luna codebase episodes are host-orchestrated depth-2 graphs. For example, the telecom source trajectory has 16
  child invocations, all with the same parent execution.

**Exposure actually earned.** `state.json`, `communication_exposure`, reports:
- communication: `graph_visits: 610` of `required_graph_visits: 864,374`;
- overall: 1,044 of 1,142,578.

The pending focus `useful-joint-focus-1102/focus.json` selects 2,048 rows:
- 968 from `t1-communication-predecessor` (the template, answer-map copy-out);
- 525 + 485 from `t1-iteration-00/01` (the template `answer_collection` iterateOn program);
- 14 delegation rows and about 56 t1-communication rows.

So most of what the model has seen or will see next is this one template.

## 2. Constructs

Weighted population estimates; the table also gives the share among the 173,330 non-template rows.

| construct | all rows | excluding the port-record template (228k rows, 56.8%) |
|---|---|---|
| eval actions | 76% | 76,981 rows (19.2% of all) |
| `nl` call in own output | 39.8% | 17,884 (4.5%) |
| awaited `nl` | 38.2% | 11,311 (2.8%) |
| per-item `nl` (`nl` plus loop/map/iterateOn) | 37.5% | 8,182 (2.0%) |
| `iterateOn` | 36.8% | 5,142 (1.3%) |
| directory reducer (opening says "This call is a directory reducer") | 38.4% | 13,745 (3.4%) |
| `folder.apply(…)` | 0.11% | ~500 (0.12%); spot checks: many are prose in `<think>` |
| `finish:true` | 5.4% | 23,980 (6.0%) |
| `decide(…)` call | 0.22% | ~900 (0.22%) |
| skills bound in opening | 0.07% | ~310 |
| skill read (`read_code("skills.x")`) | | 7 of 27 sampled skill rows; one catalog (`calculate-from-data`, `judge-against-criteria`, `move-and-verify-files`) |
| `Neuralese<T>` in a signature | 0.08% | ~300 |
| stored guidance or external native state (`external_states`) | 0 | 0 |
| decision-readout graphs (`decisions` field) | 0 of 15,703 sampled | 0 |
| tool-using coding (bash/edit/write) | 1.7% | ~7,000 |

Notes on these numbers:
- **`decide` quality.** Of 357 `decide` matches in sampled rows, several are wrong forms against the documented
  `decide(fn, ...args)`: `decide<boolean>(\`…template…\`)`, `decide(true)`, `decide(text, m, "cost/expense amount")`.
- **Root signatures.** There are 218 distinct non-template root signature names. The largest are:
  - `nl@eval:1` (16k);
  - `summarize` (3.7k);
  - `reduce_decisions` (3.2k);
  - `select_record` (3.0k);
  - `chart_depth` and `category_total` (2.5k each).
- **Coding.**
  - The `coding` family has 79,846 rows. 76,901 are `port_record_calls` rows (idiom join) in which an
    `nl@eval:8(item)` child continues a coding trajectory as a string answer.
  - Real tool-using coding:
    - t5-broad SWE-rebench: 1,843 rows (bash/read/write/edit);
    - acquired D12-D16 multi-agent SWE sessions: ~60k rows. The 45,781-row unchanged-leaf has a median of 141
      own-output tokens because its supervised target is the final report or MAST annotation JSON.
      novel-position (13,802) and bash-context (226) carry full action trajectories.
  - The 34,152 D32 rows are OpenResearcher web research (`browser.search`/`browser.open`), not coding.

## 3. Learned Neuralese and prompt-embedding operators

| operator | in the trained model | trained in `native_joint_sft_v4` | data that exercises it |
|---|---|---|---|
| 4 role-marker rows (produce/consume open/close, input and output rows) | yes (`SharedStateModel.input_rows`/`output_rows`) | yes, interface LR 1e-4 | every comm row (123,488); `marker_tok` > 0 |
| rank-8 `LowRankTokenEmbedding` on producing positions | yes, identity-initialized, `--rank 8` default | yes, LR 1e-4 | every producing span: ~123k rows, ~84% of big2 on one identical span |
| reasoning-boundary override rows | yes (`reasoning_token_ids`) | yes | all assistant turns |
| positional KV / conv / linear state splice | yes (graph executor) | yes, through backbone grads (LR 1e-6) | same comm rows; producer-consumer depth ≥2 in ~170 rows |
| decision readout (option log-loss over sibling option nodes, `data/decision_graph.py`) | mechanism only | objective configured (`decision_readout: option-log-loss`, `decision_tokens 16`) | 0 rows |
| stored native guidance / memetic soft skills (`external_states`, nzs1 blocks, `node-guidance-writer.ts`) | runtime and loader only | no; the guidance writer stage is not active (AGENTS: writers stalled on malformed producer spans) | 0 rows |
| skills with Neuralese descriptions (`renderSkillListing` accepts `NeuraleseRef`) | runtime only | no | 0; ~310 rows with text skills |
| Gaussian import channel (`model/import_channel.py`) | mechanism | no (only `rl/hindsight*.py`) | none in SFT |
| program LoRA adapters, LoRA bank/mixtures, coefficient policy, native soft states (`model/native_adapters.py`, `tune/learned/*`, `tune/soft.py`) | not in the checkpoint | no (stages 3-4 of `LFM3_RECIPE_20261028.md`) | no LFM3 tune or metalearner artifacts on disk; the old program-tuning runs were moved to `/mnt/external` on 10-29 |
| teacher text-encoder producers (`Neuralese<T>` results written through the configured writer) | yes (ordinary producer node) | yes, only through consumer loss | ~10 Luna episodes, ~19 encoder spans in sample |

Range: the directly learned operators are only the marker rows, the one rank-8 adapter and the backbone itself.

Missing relative to the plans (`LFM3_RECIPE` stage 1 lists "chains, fan-out, stored guidance, reuse"; the AGENTS
"Neuralese architecture replacement" and "Core memetic learning" sections; S6 metalearner):
- chains (a Neuralese argument relayed into a child): almost none;
- stored guidance;
- typed `Neuralese<T>` beyond about 10 episodes;
- decision readout;
- every soft-state, LoRA-mixture and metalearner operator.

Two recipe settings differ from the committed decisions, which said learned rows/adapter 1e-3 and backbone 2e-5. The
recipe runs learned rows and interface at 1e-4 and the backbone at 1e-6. Check the actual per-row update telemetry
before attributing slow Neuralese learning to data alone.

## 4. Producer quality

- **Big2, the bulk:** the producer's CE target is the full root program. The payload is a literal inside it, and the
  byte-identical producer target is scaled by 0.01. The consumer (`nl@eval:8`) receives its *instructions* as the
  imported span (`Instructions:\n<C></C>`). Its supervised output is mostly a first `item.evidence` / `read_page`
  action that hardly depends on the instruction.

  This is a constant passthrough: about 99k edges carry the same state, so nothing teaches a producer to write a
  task-specific useful state.
- **Predecessor and reversed big2:** the producer is a child whose CE target is its own answer text, which is also
  the exported span. The consumer target re-emits the answer map containing it. This teaches lossless read-back of
  imported text (a reasonable decode objective), but always in the same "answer map" frame.
- **Luna episodes:** producers are `teacher-text-encoder` nodes with no assistant output, so there is no producer
  CE. The producer encoding is shaped only by consumer gradients, which is the intended design. There are about 10
  such rows.
- **Labels:** t1-communication and loghub payloads are mostly ≤8-token labels, so the state carries little.
- **Overall:** producers are supervised only to emit text (CE). "Useful exported state" is shaped only by consumer
  loss through imports. On ~99% of comm rows that loss either does not depend on the payload (big2 instruction rows)
  or is a verbatim copy-out.

## 5. RL tasks (structure only; protected dev packet not inspected beyond structure)

The training curriculum `full-core-effects-20261102` has 61,399 cases: 271 families, 40 sources, 1,121 shapes.

| source | cases |
|---|---|
| codebase-py | 7.7k |
| natlang-codebase | 6.5k |
| reasoning-gym | 3.5k |
| about 30 external QA, table and SQL datasets | 1.1-1.8k each |

Cases by slice, mode and inline policy:

| field | value | cases |
|---|---|---|
| slice | `inline_placement` | 33.6k |
| | `nested_scoped` | 24.3k |
| | `folder_failure` | 1.8k |
| | `observation_followup` | 1.6k |
| | `iterate` | 112 |
| mode | `single_call` | 99% |
| inline | `optional` | 98.5% |

Construct coverage in the case programs:

| construct | cases | families |
|---|---|---|
| `Neuralese<string>` child results ("read each document/file/part privately, then decide from the notes", fan-in at the root) | 15,204 | 40 |
| `Neuralese` parameter (chain/relay) | 0 | |
| structured `Neuralese<T>` | 0 | |
| ≥3 `.nl` files or nested folders ≥2 deep (natlang-codebase explore families) | 5,500 | 72 |
| directory reducer (mostly `repair_type_typo` and `mbpp implement`) | 1,728 | 11 |
| `iterateOn` in the program | 80 | 5 |
| skills directories (`decision_skill_catalog`) | 12 | 1 |

The Neuralese reward bonus exists: `DEFAULT_NEURALESE_USE_BONUS = 0.1` in `rl/task_rollouts.py`.

The design is diverse in data sources and has real fan-in, but it is formulaic in operator shape. It is one
`read_X → Neuralese<string>` pattern. It has no chains, typed states, iteration over Neuralese, stored guidance or
skills.

None of it is trained now:
- `state.json`: `trained_rounds: 0`, `round: 0`;
- `task_curriculum_history` has 4 tasks, all 4/4 success with zero spread;
- `mandatory_sft_only: true`.

## 6. Gaps and ranked changes

Ranked by expected effect on the asked behaviours per unit of work. All of them use existing data, exact oracles
and self-teaching (no new teacher generation is required).

1. **Stop multiplying the template.**
   - `communication_min_visits: 7` applies to all 123k comm graphs. That is 864k of the 1.14M required visits.
     99.4% of comm rows are template edges, and ~80% carry the same instruction state.
   - The focus already promotes 1,978 template rows.
   - Change: in the same recorded objective transition, exempt identical-payload edges from the 7× quota, or cap
     readers per payload body (the D7 fix: one graph per chain, ≤64 readers per payload). Re-select the focus from
     the loghub, Qwen, Luna, delegation and t2/t3 cohorts.
   - Preserve earned exposure.
2. **Start interleaved JOLT+ RL now on the Neuralese reader tasks.**
   - Run on the 15,204 `Neuralese<string>` cases and the 5,500 deep natlang-codebase cases, with whole-graph updates
     and the 0.1 bonus.
   - Self-distil successful Neuralese-using rollouts into SFT rows. This is the cheapest path to real producer
     supervision (producers whose notes help the consumer answer) and to depth ≥3 graphs.
   - It needs `mandatory_sft_only` lifted or interleaving enabled. At 1,044 of 1.14M visits, waiting for SFT
     completion postpones outcome learning indefinitely.
3. **Convert the prepared diverse communication sources.**
   - rest27: 39,134 LFM3 pairs with 16,585 distinct payloads over 101 families, never verified or graphed (audit
     10-31 §4.6a).
   - e2e mining of verified teacher trajectories into whole-graph chains.
   - HotpotQA hop chains.
   - These give non-constant payloads and multi-hop fan-in.
4. **Add chain and typed operators in RL and SFT.**
   - Add curriculum variants with a `Neuralese<T>` argument relayed through a child (`summarize → refine(note:
     Neuralese<string>)`) and structured `Neuralese<{…}>` results.
   - Add `iterateOn` over a Neuralese state (the `priorNotes: Neuralese<string>` shape seen in 170 sampled rows).
   - Today chains are 0 RL cases and fewer than 200 SFT rows.
5. **Feed the decision readout.**
   - It has 0 rows. Build `decisions` graphs from existing labeled choice sources (CLINC, banking77, MASSIVE, the
     `teacher_decision_labels_*` worlds) with exact gold.
   - Fix or quarantine the malformed `decide(...)` targets before they teach wrong forms.
6. **Increase iteration and folder structure outside the template.**
   - Non-template `iterateOn` is 1.3% and `folder.apply` is near 0. Grow the RL `iterate` slice (112 cases) and
     `folder_failure` (1.8k).
   - Admit the `iterate_*`, `folder_edit` and `recurrence_inline_graphs` curricula fully. The idiom join shows them
     as the source of most deep rows.
7. **Native guidance and soft skills.**
   - There is 0 data for them, and the writer stage is not active.
   - Repair the malformed-producer-span stall and then collect self-authored guidance over multiple invocations of
     the same program (memetic lane), with outcome measurement on separate worlds.
   - Add skills whose descriptions are `Neuralese` references. Today skills are one 3-entry catalog in about 310 rows.
8. **Coding.**
   - Relabel the 76.9k `port_record_calls` coding rows as template rows for balancing.
   - Add the SWE-rebench v2 sample (per-repo 12) and full action trajectories (not final-report-only) from the
     acquired D12-D16 sessions.
   - Count tool-using coding separately from string-answer continuations.
9. **Learning-rate check.** Verify the interface and adapter per-row update telemetry at LR 1e-4 and backbone 1e-6,
   against the committed 1e-3 / 2e-5, before concluding that the Neuralese interface is learning from the data it
   does see.
10. **Metalearner, LoRA bank/mixtures and native soft states.**
    - These are correctly deferred to stages 3-4 on a trained backbone.
    - No LFM3 artifacts exist yet. Keep the machinery prepared, but none of these operators is "directly learned"
      today.
