# LFM3 production SFT mixture: diversity, balance and formulaic data (2026-11-03)

Owner questions:
- Where are we falling short on diversity?
- Is the mix right?
- Can we make sure the SFT data and RL tasks aren't formulaic?

Scope: read-only audit. No data, manifest, recipe or checkpoint was changed.

## Short answer

The mix is not right, and the biggest problem is not where the old audit put it. Two scripted wrapper generators
carry most of the loss and most of the compute:
- the big2 port-record communication graphs;
- the `port_record_calls` dataset-answer wrapper.

The ~140 genuine natlang teacher curricula, which teach the recipe's idioms, get about 5.6% of the loss. The loss
weighting amplifies the problem, because the `port_record_calls` research wrapper is counted as "natlang" and so
receives natlang's 4.72x family scale.

Headline numbers. Loss shares use the live quota-aware objective (`own-outputs/1`, `declared/1` family scales, 7
communication visits). Bounded sample of 23,567 rows, calibrated to the exact own-output census to within 4% per
family.

| what | rows | row % | loss % (quota-aware) | compute % (planned forward tokens) |
|---|---|---|---|---|
| A. big2 port-record communication (scripted replay, 2 nodes) | 122,700 | 30.6 | **36.9** | **65.5** |
| C. `port_record_calls` wrapper (`nl@eval:8(item)` "Do what item.request asks…") | 105,562 | 26.3 | **41.3** | 4.1 |
| F. acquired D32 OpenResearcher web sessions | 85,477 | 21.3 | 6.5 | 21.1 |
| D. Qwen/Flint boot chains (multi-node natlang) | 6,572 | 1.6 | 4.9 | 2.5 |
| H. natlang teacher curricula (~138 families) | 69,583 | 17.3 | **5.6** | 1.9 |
| everything else (SWE, D12-D18, genuine comm chains, Luna) | 11,738 | 2.9 | 4.8 | 4.9 |

The owner explicitly said a "generic dataset-answer wrapper is insufficient". In the current recipe, such wrappers
plus a single scripted communication template take **78% of the loss and 70% of the compute**.

## 0. Inputs and method

**Inputs**
- Source manifest: `ops/semantic-ticket-quality-successor-1102/train-sources.json`. It has 116 entries; one is a
  nested 67-leaf manifest, giving 182 leaves. `zstd | wc -l` counts **401,623 rows**, which equals the row count of
  `families.json`.
- CE policy: `.../ce-policy.json`, objective `own-outputs/1`:
  - token mass, `reasoning_share` 0.1, context weight 0;
  - family scales: natlang 4.7247, native-communication 1.8688, coding 0.3950, general-agent 0.1801;
  - one decision factor of 0.01 on the dominant big2 producer target `6562e294…`.
- Quota-aware basis: communication minimum of 7 visits (`quota_aware_basis`).
- Other inputs: the focus file, `big2-restoration-v2/program-template-audit-v1.json`, and the full logical-features
  census of the 176,977 joint-v4 rows (`runs/lfm3-mixture/joint-v4-quality-corrected-20261031/mixture/logical-features-{0,1}`).

**Sampling**
- Every leaf was sampled evenly: min(n, 120 + 3% of n, 6,000) rows per leaf, 23,567 rows in total.
- Each row's own-output mass was computed with the production `own_output_supervision.graph_weights` under the live
  objective, including the decision factor.
- Per-family sums reproduce the exact census (`acquired-401685-direct-corrected-v2-1102/objective-census.json`) to
  within 0.96-1.00x. All loss shares below use those calibrated masses.

**Text and duplicate measures**
- Text: own-output turns and the first user/instruction turn of every node, decoded with the pinned LFM3 tokenizer.
- Exact normalized-text duplicates: lowercase, digits mapped to 0, whitespace collapsed.
- Skeleton duplicates: string literals masked as well.
- Near-duplicates: 64-permutation MinHash over 5-word shingles, LSH with 16 bands of 4, threshold Jaccard ≥ 0.8.
- Copy ratio: the fraction of own-output 8-token n-grams that already appear in the row's context.
- Duplicate rates are **within-sample** (~3-7% of each source), so they are lower bounds on population repetition.

**Scripts and outputs (DGX):** `/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/audit-mix/`
- `sample.py`, `features.py`, `analyze.py`, `supp.py`;
- `out/analysis.json` (per-label table), `out/generators.json`, `out/rows-compact.jsonl` (per sampled row),
  `out/features.json`.

## 1. Composition

### 1.1 By logical family

Exact census plus live scales:

| family | rows | unscaled own-output mass | mass % | scale | loss % one-pass | loss % quota-aware (7 comm visits) | declared target |
|---|---|---|---|---|---|---|---|
| natlang | 104,554 | 42.7M | 10.7 | 4.72 | 48.0 | 41.1 | 41.2 |
| native-communication | 123,477 | 14.7M | 3.7 | 1.87 | 6.9 (x7 visits → 29.4 of mass) | 39.2 | 39.4 |
| coding | 79,846 | 153.3M | 38.3 | 0.39 | 15.2 | 13.1 | 12.4 |
| general-agent | 93,746 | 189.8M | 47.4 | 0.18 | 8.5 | 6.6 | 7.0 |

The family-level objective hits its declared targets. The problem is what sits inside each family.

### 1.2 By actual generator

Calibrated estimates from the sample:

| group | labels | rows | row % | fwd tokens/pass | own-output mass % | loss % one-pass | loss % quota-aware | compute % (planned, with visits) |
|---|---|---|---|---|---|---|---|---|
| A big2 port-record comm (scripted) | 2 | 122,700 | 30.6 | 1.76B | 3.5 | 7.9 | 36.9 | 65.5 |
| B other comm (teacher chains, Luna episodes) | 10 | 777 | 0.2 | 0.08B | 0.2 | 0.5 | 2.3 | 2.9 |
| C `port_record_calls` wrapper (coding + research) | 3 | 105,562 | 26.3 | 0.78B | 41.4 | 62.2 | 41.3 | 4.1 |
| D Qwen/Flint boot chains (multi-node natlang) | 1 | 6,572 | 1.6 | 0.46B | 1.3 | 7.4 | 4.9 | 2.5 |
| E SWE-rebench OpenHands | 1 | 1,843 | 0.5 | 0.05B | 3.6 | 1.7 | 1.1 | 0.3 |
| F acquired D32 OpenResearcher | 1 | 85,477 | 21.3 | 3.97B | 43.4 | 9.7 | 6.5 | 21.1 |
| G acquired other (D12-D18) | 7 | 9,027 | 2.2 | 0.32B | 5.0 | 2.1 | 1.4 | 1.7 |
| H natlang teacher curricula | 138 | 69,583 | 17.3 | 0.36B | 1.5 | 8.4 | 5.6 | 1.9 |
| Luna deep/regrade/designbrief ordinary | 4 | 91 | 0.02 | ~0 | ~0 | ~0 | ~0 | ~0 |

Inside the families:
- **natlang family:** 74.5% of the effective loss is `port_record_calls:research` (28,202 rows), 11.9% is boot
  chains, and **13.6% is the ~138 real curricula**.
- **coding family:** 96.7% of rows (77,243) are `port_record_calls:coding`. Only 1,843 SWE-rebench trajectories
  and ~2.6k acquired D13/D14 sessions are real coding work.
- **communication family:** 99.4% of rows are big2 (117,789 qualified-unique, 4,911 predecessor edges). Only ~777
  rows are genuine multi-node teacher communication chains or Luna episodes.

### 1.3 Compared with the recipe intent

The stated intent (`LFM3_RECIPE_20261028.md`) for natlang is:
- 35% semantic delegation;
- 20% exact computation;
- 15% structural iteration;
- 15% scope and child behaviour;
- 15% broader verified natlang;
- plus about 10% coding and Neuralese spread across the categories.

The joint-v4 categories present now (logical-features census over the 69,602 non-wrapper natlang rows that carry a
category):

| category | rows | share of these rows | recipe intent |
|---|---|---|---|
| scope | 36,729 | 53% | 15% |
| broad | 14,042 | 20% | 15% |
| exact | 10,833 | 16% | 20% |
| delegation | 6,130 | 9% | 35% |
| iteration | 1,868 | 3% | 15% |

Semantic delegation, the largest intended share (35%), is 9% of these rows, and structural iteration is 3%. Its loss share is far smaller still,
because those rows have little own-output mass (46-100 reply tokens per row, against 1,178 for the research
wrapper).

The intended "~10% coding" is met nominally (13.1%), but it is 82% wrapper. SWE plus acquired coding sessions come
to about 2.6% of the loss.

### 1.4 Realized exposure is decided by schedule order, not by the recipe

Learner window: `earned84-joint-r38-a4-window64-1103/learner/train.jsonl`, steps 85-179.
- 396 graphs in 4,715 s, about 300 graphs/h.
- 348 of the 396 (88%) were communication graphs.
- At this rate the quota-aware epoch (401,623 + 6 x 123,477 = 1.14M graph visits) takes about 160 days.

Consequences:
- The interleave (`sustained-packed-batch-proportional/1`) is proportional to owed visits. Communication owes 864k of
  1.14M visits (76%), so any practical horizon is dominated by big2 graph visits and big2 forward compute.
- The 2,048-row focus (1,024 edges + 1,024 `t1-iteration` teacher rows) is the only explicit diversity lever in the
  near term.

## 2. Diversity

### 2.1 Per group

Within-sample measures:

| group | n | distinct call signatures | top-signature share | distinct action sequences | top action-seq share | exact-dup targets | skeleton-dup targets | near-dup targets | identical instruction | distinct first source group / n |
|---|---|---|---|---|---|---|---|---|---|---|
| A big2 | 4,043 | **2** | 0.85 | 45 | 0.19 | 0.67 | 0.84 | **0.95** | **1.00** | 0.86 |
| C wrapper | 4,868 | 13 (2 = 0.94) | 0.79 | **20** | **0.77** | 0.00 | **0.91** | 0.05 | **1.00** | 0.93 |
| D boot chains | 447 | 78 | 0.35 | 279 | 0.08 | 0.00 | 0.00 | 0.10 | 0.70 | 0.80 |
| E SWE | 176 | n/a | n/a | 176 | 0.01 | 0 | 0 | 0 | 0 | 1.00 |
| F D32 | 5,997 | n/a | n/a | 4,873 | 0.05 | 0 | 0 | 0 | 0.37 | 0.63 |
| G D12-D18 | 575 | n/a | n/a | 46 | 0.76 | 0.25 | 0.27 | 0.32 | 0.58 | 0.77 |
| H curricula | 6,959 | 1,616 | 0.05 | 454 | 0.17 | 0.02 | 0.03 | **0.33** | 0.52 | 0.51 |

How to read the columns:
- **A and C:** every row has the byte-identical instruction. Diversity exists only in the item payload.
- **C:** 77% of rows have the action sequence `eval > return_result`, and 91% of targets are identical once string
  literals are masked.
- **H:** genuinely varied in signatures (1,616 in 6,959). However, 33% of sampled rows have a ≥0.8-Jaccard twin
  even at a ~10% sample, so population repetition is higher.
- **H, worst families by near-duplicate targets:**

  | family | near-dup share | estimated rows |
  |---|---|---|
  | `fresh_service_recovery` | 0.88 | 3,164 |
  | `reducer_apply` | 0.85 | 487 |
  | `folder_triage_inline` | 0.83 | 440 |
  | `clef_support_routing` | 0.80 (0.73 exact) | 740 |
  | `inline_union_target` | 0.78 | 567 |
  | clinc intent | 0.70 | 7,764 |
  | `named_versus_inline` | 0.68 | 699 |
  | `folder_index_inline` | 0.67 | 320 |
  | `labeled_judgments` | 0.64 | 642 |
  | `knowledge_evidence` | 0.62 | 561 |
  | `loop_rewrite` | 0.60 | 2,673 |
  | `people_lookup` | 0.60 | 513 |

### 2.2 Template concentration (full census)

From the logical-features near-duplicate clusters over all 176,977 joint-v4 rows:
- One template prefix, `9db6dc7a88349e81` (the `port_record_calls` reducer), covers **105,532 rows = 26.3% of the
  whole source**. That is 77,243 coding rows (14,833 tasks) and 28,289 research rows (4,078 tasks).
- The next largest prefix, `0e01de7d6d44183a`, is CLINC intent with 8,205 rows.
- 1,506 clusters sit at exactly the 24-row balance cap (36,144 rows in clusters of ≥24). The cap is binding broadly.
  It limits rows per world, but not per template.

### 2.3 Graph structure

| measure | rows | row % | loss % |
|---|---|---|---|
| single-node rows | 271k | ~68 | n/a |
| 2-node producer/consumer graphs | 123,507 | 30.8 | 37.2 |
| ≥3-node graphs | 6,066 | 1.5 | 7.0 |

- Almost all 2-node graphs (122,815) are big2. Almost all ≥3-node graphs (6,044) are boot chains.
- The natlang joint-v4 rows by recorded call-graph depth:
  - 0 or 1: 50,916 rows;
  - 2-4: 20,934 rows;
  - ≥5: 7,384 rows.
- Branching is mostly 0 (118,608 of all feature rows) or the cap of 10 (14,831).
- Neuralese markers appear in 30.75% of rows. 99.7% of those are big2. Genuine teacher-chain Neuralese is 663 rows.
- `iterateOn` is in 37% of rows, and 82% of those are big2's single reducer template. Another 15% is the
  `port_record_calls` reducer. The natlang curricula and other ordinary rows contribute about 4.5k rows (1.1%).
- **Unknown coverage:** 97,743 joint-v4 rows (schema-2 features) and all 94,505 acquired rows carry no depth, branching
  or task-shape fields. Coding/research graph shape is therefore known only from the sample, which shows the fixed
  one-child reducer.

### 2.4 Same generators reskinned

- **One scripted port-record generator** feeds A (big2) and C (`port_record_calls`):
  - fixed reducer + inline child;
  - the child's identical instruction "Do what item.request asks, using only item.evidence…".

  Together: **228k rows (57%)**, **78% of the loss** and **70% of the compute**.
- **One procedural generator** (`natlang.inline_curriculum_generator/1`, with its `sNNNNN` seed groups) accounts for
  about 54% of the sampled H rows (3,772 of 6,959). The rest are authored service-recovery and bounded-decision
  worlds, plus public-dataset curricula (AG News, banking77, CommaQA, QuALITY, bgkit). The same generator also
  produces 1,369 of the 1,453 current RL cases (`cases-v3/selected.jsonl`):
  - 1,191 of those RL cases are from a single seed, `s81028`;
  - 55 of the 56 RL families also appear as SFT natlang families.
- **The owner's diverse Luna codebase generation** reaches the mix as about **100 rows**: Luna native episodes 11,
  deep/regrade ordinary 51, designbrief 37. That is 0.025% of rows and about 0% of the loss.

## 3. Formulaic data, with examples

1. **`port_record_calls` wrapper (C), 105k rows, 41% of the loss.**
   - The child `nl@eval:8(item: Item): string` gets the constant instruction above.
   - Its own outputs are: read inputs (opening turn, masked), `item.evidence`, then `return_result` with a
     JSON-escaped assistant message from an external model.
   - That message includes the external model's `"reasoning"` field. It is supervised as reply tokens, so the 10%
     reasoning share does not apply.
   - The parent `answer_collection()` runs one fixed `iterateOn` over `items/*.json`, then returns the children's
     answers verbatim. 19-20% of these rows have copy ratio > 0.5.
   - The natlang family scale makes the research half the single largest loss item: **30.6% of the total**.
   - Examples:
     - child coding row: `text:whole-trajectory:teacher-program:46a2cda170841b1dc665:decision:0008:source-sha256:a9bf1468…`
       (math-with-Python D05 message returned as a JSON string);
     - parent: `…teacher-program:fb42d8bdaef6bc44f53c:decision:0009:…`;
     - research: `…teacher-program:3de5d8925a2f61c325cd:decision:0008:…` (D01 Judah Waten sub-agent message).
2. **big2 (A), 123k rows, 37% of the loss and 66% of the compute.**
   - Producer: `answer_collection()`. Consumer: `nl@eval:8`.
   - The program audit finds one producer target `6562e294…` on 93,229 of 108,353 graphs (86%). It is
     down-weighted to 0.01, but its full 14k-token forward cost is still paid 7 times.
   - The Neuralese payload is the constant instruction string. Consumer targets are mostly paging and evidence
     reads.
   - The world groups are dominated by ALFWorld (7,405 graphs in one `alfworld-game:json_2.1.1/train` group).
   - Example: `edge:7aaa208a40d8deb1a6b9806ebe47c414c00b831a03f4f6ec02ccf7b271b5adc2`.
3. **Keyword/regex substitutes for semantic judgment, outside the reviewed ticket families.**
   - The 2026-11-02 review covered only semantic-ticket families.
   - Detector: judgment children (`nl@…` signatures) whose own code applies a regex `.test`/`.match` or a literal
     keyword `.includes`. It flags about 2,880 rows (0.7% of rows, 1.3% of the loss).
   - A manual check of 24 random hits found about 14 clear substitutes, so the population estimate is ~1,600 rows.
     The rest were legitimate exact parsing of amounts, dates and line lookups.
   - Clear cases:
     - intent labelling by regex over the option list: `chain:qwen15-v14:teacher-program:835f7257cbd397721bfc`,
       `text:teacher-program:644e26e822854510807c:action:0019`, and the banking77 labeller
       `text:teacher-program:d874dbf28189a6df38ae:…` (`/fee|charge|cost/i.test(q)`);
     - feedback labelling by `.includes("bug")`/`"error"`: `chain:qwen36-internal23:teacher-program:d7ac520f8513a9378258`;
     - same-person decisions by a surname regex: `text:teacher-program:4698f7c5bcbdf5ec698c:…:action:0054`,
       `…bb27a311166cbd8efca3…`;
     - incident judgment `/error|latency|outage|impact/i` in `nl@eval:5(…note): boolean`:
       `text:teacher-program:5298f27c6a7e19758c1d:…` (`paged_late_row`);
     - story harm judged by keyword counts: `text:teacher-program:f1a056ce391fd7f9aa3c:…:action:0010`;
     - billing support action via `/pending/i.test(s.history)`: `chain:qwen-boot2-21:teacher-program:7fbf02a24401e690bdc6`;
     - ProofWriter decided by `.includes("rabbit")`: `…8aaff514b26abb12030e…`.
   - Concentration: `story_choice` 31%, `inline_late_binding` 39%, `people_chain` 38%, `inline_structured_extract`
     30%, `knowledge_evidence` 23%, CLINC chains.
4. **Label regurgitation (D16/D18 MAST, ~3,577 rows).**
   - Task: "Read the complete observed execution trace … return its source-reported failure annotations".
   - Target: `<think></think>` followed by a 14-flag JSON. 97% (D16) and 80% (D18) of targets are exact duplicates;
     near-duplicates are 98-100%.
   - Context is 13-21k tokens per row, against 110-130 mass.
   - D18 targets literally include `"note":"synthetic anomaly-injection labels; 1.1-1.5 always 0"`.
   - Example: `text:upstream:expansion:D16:ChatDev:ProgramDev:26:MAD_full_dataset.json:26:…`.
5. **Final-message-only continuations (D12, 2,693 rows).**
   - Task: "Continue the original coding execution for X after the complete observed transcript". A whole
     38-80k-token transcript sits in one tool block; the target is only the closing summary (141 mass per row).
   - Several rows per task: `…pallets_click_task/2800/f1_f3…:57:final` and `…:58:final`.
   - The D13 sessions sampled with copy ratio > 0.5 in 80% of rows. Their final messages restate the transcript.
6. **Thin, repeated curricula.**
   - `clef_support_routing`: 73% exact-duplicate targets, mass 48 per row.
   - `fresh_service_recovery`: 88% near-duplicate. `select_record(record: Row, policy: Policy): boolean` is still the
     top signature, as in the 2026-10-31 audit.
   - Single-signature families: `loop_rewrite` uses only `category_total(): number`; `recursion_rewrite` uses only
     `chart_depth()`.
   - Teachers sometimes decide in `<think>` and hardcode the list instead of delegating per item. For example
     `teacher-program:2e7620c8f9f1aedd2696:decision:0001` (archive tickets) hardcodes `['droosh.md','sod.md']`.
7. **D32 OpenResearcher (85k rows): diverse but single-format.**
   - Single-agent web Q&A ("Question: …"): 4,873 distinct action sequences in 5,997 sampled rows, no duplicates.
   - It is one dataset, one domain and one tool protocol, at 21% of rows and 21% of compute. Chunking multiplies
     sessions: 37% of sampled rows share their question with another sampled row.
   - Average reasoning is 4,828 tokens per row, against 2,052 reply tokens.

## 4. RL tasks

Source: `cases-v3/selected.jsonl`. I did not verify that it is the live packet; it is the newest selected set.

- 1,453 cases in 56 families.
- 94% come from `inline_curriculum_generator/1`, and 82% from one seed (`s81028`).
- 75 cases (5%) mention Neuralese.
- No coding, research or acquired-world tasks.
- Slices:

  | slice | share |
  |---|---|
  | `inline_placement` | 42% |
  | `nested_scoped` | 19% |
  | `observation_followup` | 16% |
  | `folder_failure` | 14% |
  | `iterate` | 8% |

- The RL families are the same generator families the SFT curricula already train on. This matches the AGENTS.md
  record of a 20/20-correct, zero-spread JOLT batch.

## 5. Mix judgment against the goals

| goal | what it gets now | judgment |
|---|---|---|
| natlang execution idioms (per-item `nl`, exact eval, `iterateOn`, scope) | 5.6% loss (curricula) + 4.9% (boot chains) | **badly under-weighted**; the 41% natlang target is spent on the research wrapper |
| native Neuralese communication | 39% loss, 66% compute, 99% one scripted template | **over-weighted in volume, under-diverse**; genuine multi-node comm is 2.3% of loss |
| coding | 13% loss, 82% of it the wrapper; SWE/D13/D14 about 2.6% | real coding **under-weighted** |
| tool use / general agent | D32 6.5% loss, 21% compute, one dataset | compute-heavy for its loss; narrow |
| deep call graphs | ≥3-node graphs 1.5% of rows / 7% of loss; depth ≥5 natlang 7.4k rows | **under-weighted** |

## 6. Recommended changes, ranked

1. **Stop counting the `port_record_calls` wrapper as natlang, and stop it dominating the loss.**
   - Change: either reclassify research+coding wrapper rows out of `natlang`/`coding`, or apply decision factors (as
     already done for big2) to the child's `item.evidence` + verbatim `return_result` targets and to the parent's
     verbatim echo. Keep one representative row per task, not per item.
   - Impact: this frees **41% of quota-aware loss (62% one-pass)**. Rerunning the declared scales without the wrapper
     puts natlang at the max-scale cap of 8: genuine natlang (curricula + chains) rises from about 10% to **~28%**.
   - Caveat: with the current targets the communication family would then rise to 48%, almost all big2. Do item 2
     together with this one.
2. **Cap big2 by template and by visits.**
   - Change: drop the 7-visit minimum for big2 graphs whose producer matches the 0.01-factor template, or cap at 1-2
     visits; cap readers per payload (the old D7 fix).
   - Impact:
     - 2 visits instead of 7 free **~47% of planned forward compute**; 1 visit frees ~56%.
     - Together with item 1, the scales land near natlang 36% / communication 43% / coding 13% / general-agent 8%.
     - The communication share should then be refilled from genuine chains (item 4), not from big2.
3. **Remove or regenerate the formulaic acquired views.**
   - Withdraw D16/D18 MAST label regurgitation (~3.6k rows).
   - Collapse D12 "final continuation" rows to one per task, or drop them (2.7k rows, 59k context tokens each, 141
     mass).
   - Impact: under 2% of the loss, but about 2.6% of compute and a clear formulaic signal removed.
4. **Grow genuinely diverse communication and deep-graph data.**
   - Sources:
     - rest27 (39k pairs, 101 families, 16.6k distinct payloads; still unadmitted per the 10-31 audit);
     - teacher-mined chains (B is only 777 rows now);
     - view/ask and intermediate-value builders.
   - Impact: genuine communication is currently 2.3% of the loss. Moving it to 10-20% is the only way to "instill
     Neuralese" without training on one template.
5. **Extend the semantic-substitute review corpus-wide.**
   - Run the existing `semantic_ticket_code_check.mjs` criterion over every `nl@…` judgment child in all natlang
     families and the boot chains, not only ticket families.
   - Impact: about 1,600 rows (around 0.7% of the loss). Small in mass, but these teach exactly the anti-idiom the
     owner flagged, inside the highest-weighted family.
6. **Re-weight inside the curricula.**
   - Change: cap per template signature (for example `select_record`, `category_total`, `chart_depth`, the CLINC
     reducer), not only per world. The existing cap is 24 rows per near-duplicate world cluster. Bound the CLINC
     family (7.8k rows, 70% near-duplicates).
   - Impact: modest loss shift; better per-step diversity once item 1 raises the curricula's weight.
7. **Admit actually diverse Luna codebases and the prepared losses.**
   - Luna codebases are about 100 rows now.
   - The 10-31 recovery list still appears absent by id prefix: Search/OWL D02/D21, native-sft-v13-r2,
     static-s7k, evidence-terminal, rest27, S3 45-family port records.
   - Only codeagent arrived, and it arrived as the wrapper.
   - Admit them with multiple program shapes, not one reducer.
8. **D32: keep, but bound compute.**
   - It is the only broad tool-use source, but it costs 21% of compute for 6.5% of the loss.
   - Cap visits per question/session, for example the first N chunks, and dedupe shared-question chunks.
9. **RL: diversify the task supply before more JOLT rounds.**
   - Add families from other generators and seeds, coding/research worlds with oracles, and more of the
     Neuralese-declaring tasks (5% now).
   - Hold back RL families already saturated in SFT.

## 7. What I could not measure

- **Realized per-row exposure.** It lives in the 15 GB checkpoint, which I did not load. Realized mix is inferred only
  from the 94-step `train.jsonl` window (88% communication graphs) and the step-67 quota audit.
- **Semantic diversity beyond surface measures.** MinHash and skeleton hashing detect surface repetition only.
  Distinct strings can share a behaviour (C), and similar strings can hide distinct tasks.
- **Population duplicate rates.** All duplicate rates are within-sample (~3-7% per leaf), so they are lower bounds.
  The full-census numbers are the logical-features clusters (§2.2) and the big2 program audit.
- **Graph depth and branching** for the 97,743 schema-2 joint-v4 rows and all 94,505 acquired rows. They have no
  feature fields; I counted nodes only.
- **The regex/keyword detector.** It is heuristic, and its true-positive rate (~58%) comes from a 24-row manual check.
- **The decision-readout loss term.** It is outside the token census, and its contribution is not included.
- **RL packet.** The live RL packet was not confirmed; cases-v3 is assumed to be representative.
