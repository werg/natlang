# Diversity end to end: where repetition enters, what it measures, what caps it (2026-11-25)

Owner: "I don't want to train on repetition at all" (2026-11-24); "truly counteract all that repetition throughout
and make sure everything end to end is moving towards diversity" (2026-11-25). DECISIONS 2026-11-25 "Diversity end to
end" records the change; this file is its inventory.

## One mechanism

`training/neuralese/natlang_neuralese/common/diversity.py` holds the measures and the caps that every stage uses.
Before this change there were separate copies in `grpo.py`, `measure_repetition.py` and `sft_sources.py`.

- **Shape key.**
  - `program_shape` is the program with item data abstracted. It moved here from `measure_repetition.py`, which now
    imports it.
  - `task_shapes` merges base shapes that share a file skeleton, services and mode and whose abstracted texts have
    word-trigram Jaccard >= 0.7. This fixes a known error: lowercase item words inside template prose used to split
    one template into many shapes.
  - Examples: `codebase-ts-callers` "…calls something named estimate/best/compile…" is 117 base shapes but one
    template; `git-count-commits` is 28 base shapes but one template.
  - `id_shape` covers a unit without a program row. Its id has hex and digit runs abstracted when words remain (a
    generator template). Otherwise the id stands as its own shape (content-hash ids, such as operators).
- **Near duplicates.**
  - `NearDuplicates` is a 64-permutation MinHash with LSH of 16 bands × 4. It moved from `sft_sources.py`.
  - Text uses word 5-shingles. Targets use token-id 5-grams (`token_signature`, vectorized).
  - A unit with fewer than 64 target tokens has no content signature (decision readouts, one-word answers).
- **Caps.**
  - `cap_shapes` reorders an order without dropping anything. It moved from `grpo.py`.
  - `ShapeQuota` is a share of a total, counted in items or tokens.
  - `drop_near_duplicate_units` keeps the better source's copy.
  - `round_robin_by` puts each task once before any task twice.
- **Report.**
  - `diversity_report` gives, per part: units, mass, distinct shapes, effective shapes (exp-entropy of mass),
    top-shape share, top-10 share and near-duplicate share.
  - `regression_flags` raises three flags against the previous build:
    - effective shapes below 0.8× the previous value;
    - top shape above 1.5× the previous value and above 2× the cap;
    - near-duplicate share above 5%.
  - CLI: `python -m natlang_neuralese.common.diversity mix BUILD [--cases POOL ...]`.

## Inventory (stage, code, what limited repetition before, measured, change)

Measured on real data. The pool is the registry role `grpo-pool`: 37,836 train rows. The mixes are aei1 r0007 `mix/`
and r0008 `mix.repeat-aside-1795543970/`. Held is held-v3.

### 1. Task pool admission

**Code:** registry role `grpo-pool`, `grpo.load_tasks` (`scripts/neuralese_data/grpo.py`), the source builders under
`scripts/rl_tasks/sources/`.

**Limits before:**
- `measure_repetition.py` and `simulate_selection.py` are run by hand before a batch is admitted.
- Nothing at admission caps shapes per source.

**Measured:**
- Base shapes: 6,637 distinct, 1,272 effective.
- Merged shapes: 4,043 distinct, 828 effective. The top shape is 2.6% of the pool, the top 10 are 13.1%.

Per source (rows, base → merged shapes, merged top share):

| Source | Rows | Shapes (base → merged) | Top share |
|---|---|---|---|
| natlang-codebase | 2,683 | 1,031 → 152 | 36.5% |
| codebase-py | 2,822 | 357 → 111 | 8.2% |
| codebase-natlang | 988 | 303 → 42 | 18% |
| git-history | 736 | 322 → 35 | 8.7% |
| decisions | 2,524 | 1,103 → 648 | 4.2% |
| bgkit | 7,263 | 126 → 117 | 6.5% |
| composed | 6,493 | 145 → 112 | 4.4% |
| authored | 4,984 | 2,617 → 2,480 | 0.2% |
| reasoning-gym | 1,770 | 3 → 3 | 35% |
| math | 763 | 3 → 3 | 37% |
| babi | 620 | 2 → 2 | 55% |
| invented | 271 | 21 → 21 | 14% |

Every tool or QA dataset wrapper has 2–7 shapes.

**Change:** none at admission. The rows stay in the pool (owner: no procedural generation is added, and existing
rows are not dropped). Every consumer caps by the merged shape instead. `measure_repetition.py` now reports merged
shapes (`distinct_shapes`, `effective_shapes`, `top_shape_share`) beside the base shapes (`distinct_base_shapes`).

### 2. Collection selection

**Code:** `expert_iteration.selection_order` (`scripts/neuralese_data/expert_iteration.py`, `selection_order` and
`select`) and the GRPO walk `grpo.ExploreWalk`.

**Limits before:**
- Tiers, then family round-robin, then `cap_shapes` at 1% of any prefix (c23f8b61a).
- GRPO's learnable cycle was a plain reshuffle (`Collector._pick`).

**Measured:** DECISIONS 02:50, 2,000-task prefix: top shape 3.45% → 1.0%, effective shapes 876 → 1,131.

**Change:**
- The cap now uses merged shapes (`grpo.task_shapes` → `diversity.task_shapes`), which closes the
  lowercase-item-word gap.
- GRPO's learnable (in-band) cycle is now shape-capped like the walk (`_pick`, under `--explore-shape-cap`).

### 3. Samples per task and own successes per task

**Code:** `expert_iteration.py` collect (`--group str(args.k)`), `sft_sources.capped` with `core_cap`.

**Limits before:**
- One group of k per task per collection.
- Each new policy version re-queues band, failed and unmeasured tasks.
- In the mix, own successes per task are capped by `round(1 + (cap-1)(1-p))` with cap 4, or 8 for a held construct.
  Then a family share of 5%, a wrapper share of 15% and `--max-prior-exposures 1`.
- Own successes are near-duplicate-checked by action MinHash at 0.8.

**Measured:**
- r0007 rl-own has 834 units over 343 merged shapes (236 effective).
- The top shape is 3.6% of rl-own mass: `bbfbeb328e4b1cc4`, a reasoning-gym wrapper. r0008 is 2.9%.

**Change:** `capped` adds a shape share after the family share (`--max-shape-share`, default
`diversity.MIX_SHAPE_SHARE` = 2%, floor 4). One template spanning 57 families is now held like one family.

### 4. Neuralese task variants

**Code:** `scripts/rl_tasks/neuralese_variants.py` (`variant`), `grpo.load_variants`, `Collector._new_group` (swaps
in the variant for half of a task's groups).

**Limits before:**
- A variant does not double a task's groups.
- But it is another task id, so the SFT per-task cap, the per-task token cap and the aux `task_tokens` counted the
  original and its variant separately. One item could reach 2× its allowance.

**Change:** the build maps every task to its item (`curriculum.external.variant_of`). The per-task cap in `capped`,
`--max-task-target-tokens` and every aux part's `task_tokens` now count a variant together with its original.

### 5. Teacher campaign goal order

**Code:** `teacher_gen.order_tasks` (`scripts/neuralese_data/teacher_gen.py`).

**Limits before:** gap tiers, then family round-robin.

**Measured:**
- teacher-r0008 order, 2,000 prefix: top shape 3.15%, 397 effective shapes.
- teacher-r0007: 2.35%, 564 effective.
- The 894 tasks attempted in r0008 cover 506 shapes, top 1.1%.

**Change:** `cap_shapes` runs after `order_tasks` (`--shape-cap`, default 1%; `inputs.json` records it). The 2,000
prefix becomes top 1.0%, 455 effective (r0008) and 601 effective (r0007). This takes effect at the next campaign work
directory, because `tasks.jsonl` is written once per work.

### 6. Filled-reasoning order

**Code:** `filled_reasoning.ordered` and `teacher_episodes`.

**Limits before:**
- Shuffled, failing tasks first, preferred works first.
- No per-task spread: a task's several teacher episodes could fill one after another.
- No shape cap.

**Change:**
- Within each group (preferred works, the rest), every task gets one fill before any task gets a second
  (`round_robin_by`).
- `cap_shapes` at 1%, using the episode's `shape` from `teacher_episodes` (`--shape-cap`; 0 opts out).

### 7. Filled units in the mix (the largest loss share: 43% in r0008)

**Code:** `sft_sources.build` `take_unit` (aux, `replaces=teacher`).

**Limits before:**
- A filled unit stands in for its teacher episode.
- But it was admitted whatever the teacher selection did. A teacher episode dropped as a near duplicate or over a
  task, family or wrapper cap still trained through its fill. r0008 funnel: 416 near duplicates, 6,127 over family
  cap, 1,102 over wrapper share.
- The only cap on fills was 40k tokens per task.

**Change:** `--aux-teacher-selection follow`, the default. A fill whose teacher episode the build left out is left out
with it (`teacher_not_selected`). `ignore` opts out.

### 8. Every aux part: operators, fallow22, neuralese-graph, own Neuralese graph, live, subagent→Neuralese

**Code:** `sft_sources.build` (`load_aux`, `take_unit`). Builders: `operator_rows.py`, `broad_graph_october.py`,
`sft_sources.neuralese_graph`, `subagent_neuralese.py`.

**Limits before:**
- In the mix: the part's share, per-task tokens, exposure, and one (task, round) across parts.
- No near-duplicate check against core episodes or other parts.
- No shape cap.

Builder-side limits before:

| Builder | Limits |
|---|---|
| operators | merge keeps the first N per source; no dedup |
| fallow22 | one per task, MinHash 0.8; family and wrapper caps off; FALLOW_REGISTER: root eval is 19% of mass from 8 templates |
| neuralese-graph | 2 per task |
| subagent | one per exact normalized question |

**Measured** (final r0007 mix):

| Part | Units | Shapes | Effective | Top | Near-duplicate share |
|---|---|---|---|---|---|
| operators | 1,723 | 1,723 | 1,330 | 0.3% | 0.6% |
| fallow22 | 712 | 712 (id shapes) | 690 | — | 0.7% |
| neuralese-graph | 44 | 35 | 31 | 6.9% | — (top 10 46%) |

**Change:**
- Each aux part gets a shape quota. It is `--max-shape-share` of the part's token budget for share-bounded parts, or
  of its units for `share=all`. Shapes come from the task's pool row, else `id_shape`.
- After all parts are assembled, a near-duplicate pass over target token 5-grams at `--dup-threshold` (0.8) drops a
  unit that repeats a kept one. Priority order: rl-own, jolt-guided, teacher, filled, then the rest.
  `--mix-near-duplicates keep` opts out. This also closes a latent bug: the October path referenced an undefined
  `dedup`, and now uses the build's index.
- Builders are unchanged. The mix enforces the rules on whatever they produce. subagent→Neuralese gets the same rules
  when it enters as an aux part.

### 9. The mix build's per-task, per-family and per-wrapper caps

**Code:** `sft_sources.capped`, `wrapper_capped`, `student_selection`, `breadth_first`.

**Limits before:** per task, family 5%, wrapper 15%, breadth-first over tasks, exposure 1.

**Measured** (whole mix, units by merged shape):

| Mix | Units | Shapes | Effective | Top | Top 10 | Near-duplicate share |
|---|---|---|---|---|---|---|
| r0007 | 11,803 | 6,471 | 1,526 | 2.1% | 8.1% | 0.04% |
| r0008 | 11,665 | 6,456 | 1,554 | 1.7% | 7.7% | 0.03% |

Teacher r0007: 1,482 shapes, 755 effective, top 2.3%. Filled r0007: 2,666 shapes, 1,105 effective, top 1.4%.

**Change:**
- The shape share described in item 3.
- `mix.diversity` (the report), `mix.diversity_flags` (against the newest `--prefer-unseen` build that has a report,
  or `--diversity-baseline`) and `target_cap.near_duplicates` in `build.json`.

### 10. Near duplicates across rows, reports and targets

**Limits before:** core episodes only (action-text MinHash 0.8). The aux parts and anything crossing parts went
unchecked.

**Change:** the cross-part target pass in item 8. Reports and other prose-only side artifacts are not training
targets and are not checked.

### 11. Held sets: siblings and shape overlap

**Code:** `held_eval.exclusion_ids` and `training_exclusion_ids` (`training/neuralese/natlang_neuralese/rl/held_eval.py`):
ids, `split_group`, `source_groups` and every `held-eval*` role.

**Limits before:** no shape-level check.

**Measured:**
- 177 of held-v3's 256 tasks share a merged shape with training-pool rows: 157 of 235 distinct held shapes, 7,235 of
  36,282 training rows.

**Change: none, deliberately.**
- Held-v3 is drawn i.i.d. from the pool's sources. A shared template with other items is the in-distribution
  generalization it measures.
- Excluding those shapes would remove a fifth of the pool, including the whole of several dataset wrappers.
- Sibling closure (groups) stays the leakage rule.
- The number is recorded here so that a held gain concentrated in shared-shape tasks can be read against it.

### 12. Program-inventor campaign

**Code:** `scripts/rl_tasks/program_atlas.py` (`take`, `claim`, `near`, `bank_check`) and
`scripts/rl_tasks/sources/invented.py` (`admit`, `stage_cases`, `publish`).

**Limits before:**
- Cards are TF-IDF near-duplicate checked at 0.35 against accepted ideas, ideas being built, the repo and the pool.
- Accepted ideas feed back (`accepted.jsonl`).
- Stage tasks: one per distinct leaf judgment, exact `(path, args)` dedup within a case, no cap per program.

**Measured:** the invented pool has 271 rows over 21 shapes, top 14%.

**Change:** none in the campaign. Stage tasks are pool rows, so collection's shape cap (item 2) and the mix's shape
share (item 3) bound them like any template.

### 13. Aux shares against the actual mix (coordinator, r0009)

**Code:** `sft_sources.build`, step 3 (`total_tokens`, `aux_state` budgets), `trim_shares`.

**Before:** every share-bounded part's budget was share × (teacher actions + every *loaded* unit of the share=all parts)
/ (1 − fixed shares). r0009's filled held 15,480 units and took 470 (the rest over-exposed), so budgets were set on a
53.8M total while the mix had 11.6M: operators (share 0.05) carried 23.1%, own-neuralese-graph 8.1%, fallow22 4.6%.

**Change (`--aux-share-of actual`, default; `nominal` opts out):** share=all parts are taken first and counted by what
they took; `trim_shares` then holds each share-bounded part to its share of the mix as it is. The own share stays a
floor at the old rule's budget. The empty `exposure_cap.parts` was a reporting gap (exposure is filtered at the
candidates); `exposure_cap.candidates_over_exposed` now reports those counts.

**Trial on r0009's inputs:** 6.97M (r0009 11.64M); operators 2.9% (23.1%), own-neuralese-graph 2.9% (8.1%), fallow22
0.06% (4.6%), rl-own 63.2% (4.40M; r0009 4.97M), teacher 20.3%, filled 7.9%; 3,037 units, 1,702 shapes, 487 effective,
top 1.6%.

## Not checked

- Effective shapes measure declared program text, not episode behaviour.
- Programs that differ only in abstracted identifiers still merge.
- The 0.7 merge threshold was checked on the largest merges, by reading the examples. It was not calibrated against
  audited counts.
- subagent→Neuralese is being rebuilt and was not measured.
- Operator and fallow22 builders were not changed.
- The GRPO learnable-cycle cap has no unit test of `Collector._pick`. It reuses `cap_shapes`, which has tests.
