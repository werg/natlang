# GRPO task supply, 2026-11-17: new writing, chat, reasoning and agentic families (task-supply agent)

Question: which natlang behaviours does the GRPO pool exercise rarely, and what qualified tasks fill those gaps for the
next pin (grpo-v4; later batches feed later pins)? Method: measure the pool and grpo-v3-run1's own groups, pick gap genres,
author task programs with exact oracles (Haiku authors, kit rounds 3-8 rules), qualify each by reference, negative and
alias replay on the grpo-v3 pin runtime, gate repetition against the pool, blind-solve every batch with an independent
Sonnet solver, admit, and revive one fallow verified source. Kit: `RL_TASK_AUTHORING_KIT.md`; round brief
`/mnt/external/natlang-rl-authoring/AUTHOR_BRIEF_ROUND9.md`; work files under `/mnt/external/natlang-rl-authoring/round9/`.

## 1. Gap analysis of the current pool

Pool = grpo.py `POOLS` (memetic-program-source-20261105 + bgkit-kr-20261103): 33,164 rows, 321 families (31,307 eligible
in run1's `pool.json` after the held-v3/v2 closure and errata).

By genre (scratch bucketing of `source`/`family`):

| genre | rows | share |
|---|---:|---:|
| reading/QA over documents (bgkit readers, composed Neuralese readers, multihop QA) | 14,246 | 43.0% |
| codebase questions and repairs | 6,794 | 20.5% |
| math, puzzles, logic | 3,866 | 11.7% |
| decision desks (routing, verdicts, design-brief gates) | 2,704 | 8.2% |
| authored operations desks (Haiku rounds 1-7) | 2,269 | 6.8% |
| tables, SQL, finance QA | 1,485 | 4.5% |
| inline curriculum (services, folders) | 1,454 | 4.4% |
| single-call tool/function selection (ToolACE, xLAM, NESTFUL) | 346 | 1.0% |

- **Writing**: 32 rows (0.1%, one inline `curriculum_edit_stream` family). No editing, formatting, templated
  correspondence or redaction task anywhere else.
- **Chat / conversation**: 0 rows. The closest family (`auth_confirmed_dialogue_state_updates`, 93 rows) is one
  customer and one agent confirming a booking.
- **Multi-step agentic work against a stateful service**: 0 rows. Services in the pool are read-only repos (3,810),
  SQL/table snapshots, single-call API receipts (346) and inline-curriculum toys (~900). The verified tau-bench store
  sources (`build-v3/{tau-retail,apigen-mt-retail,apigen-mt-airline}`, store service with a final-state digest oracle)
  were built and verified on 2026-10-31 but never assembled into any pool (fallow).
- **Every authored family is an operations "desk"** over an invented ledger (35 families: diagnosis desks, rota repair,
  reconciliation, loan desks...). Different domains, one genre.
- Construct shares (`measure_repetition.py`, `round9/repetition/pool.json`): skills 1.5%, decide calls 2.5%,
  effect-state oracles 9.5%, folder reducers 8.1%, iterate 12.8%, Neuralese results 51.6%.
- Repetition: 8,547 distinct programs (1,405 effective), 7,773 instruction texts (1,318 effective), 173 structures (34.9
  effective), top structure 19.6%.

Learnability (run1 `ops/grpo-v3/run1/groups`, 566 scored groups on 370 tasks through step 10): 70% of groups had reward
spread, 13% all-fail, 17% all-pass. By source: authored rows 0.75 spread with only 5% all-pass (the least saturated
large source); decisions 0.73; bgkit 0.71 (21% all-fail); composed 0.68 (20% all-pass); reasoning-gym 0.51 (41%
all-pass); math 0.12 (88% all-pass). Base pass rates of authored tasks (`exp1-rft/base-pass-rates.json`) are low: mean
0.28 at level 2 with 59% never solved, so round-9 authors were told to write somewhat easier programs (about 30% level 1,
45% level 2, 25% level 3; at most 3 helpers; clear surface cues).

## 2. Families

Two parts: authored round-9 families in the new genres, and a revived store-service source.

**Authored, round 9.** Haiku 5.5 authors w175-w204 wrote one family per batch. A first batch opens a family and an
extension batch adds 8 or more new programs to it. A Sonnet blind solver checked every batch. The round opened 16
families in 31 batches. 709 tasks were admitted to `natlang-rl-authoring/admitted`: 489 train, 116 dev and 104
confirmation by the authored world split. 45 were held back.

| family | genre | admitted | train | batches |
|---|---|---:|---:|---|
| manuscript_revision_desk | writing: apply/resolve editor comments, style skill, undo, renumber, merge reviews | 43 | 33 | w175, w191 |
| citation_style_formatting | writing: typed reference extraction, style skills, dedup, ibid, BibTeX, footnotes | 51 | 38 | w176, w197 |
| document_redaction_desk | writing/privacy: redaction policy skill, pseudonyms, masking, retention, erasure | 43 | 24 | w182, w198 |
| mail_merge_letters | writing: templated letters, conditional paragraphs, suppression, reprint diff | 27 | 15 | w185 b0001+b0002 |
| glossary_terminology_pass | writing: terminology glossary, sense readouts, length fit, translation memory | 24 | 22 | w186 |
| recipe_and_method_rewriting | writing: scaling, unit skills, substitutions, dilution, reorder, hazard notes | 49 | 27 | w188, w202 |
| contract_redline_desk | writing: playbook skill, redlines, renumbering, defined terms, side letters | 49 | 33 | w190, w204 |
| release_notes_compiler | writing: change records to release notes, reverts, semver, advisories | 48 | 30 | w183, w199 |
| minutes_action_followups | writing/summarization: action registers, two-meeting diffs, agendas, motions | 43 | 28 | w179, w194 |
| survey_free_text_coding | writing/annotation: codebook skills, two-coder reconciliation, trends | 48 | 39 | w189, w203 |
| chat_moderation_desk | chat: cue-list readouts, thread cascades, appeals, rate limits, sanction ladders | 47 | 37 | w177, w192 |
| canned_reply_composer | chat/writing: intents, phrasebook snippets with slots, escalation, audits | 47 | 31 | w178, w193 |
| group_chat_plan_extraction | chat: many-speaker polls, potlucks, carpools, IRV, cost splits, rota swaps | 47 | 31 | w180, w195 |
| faq_answer_desk | chat: FAQ matching, slot fill, verified gates, follow-ups, version notices | 48 | 36 | w184, w200 |
| chat_handoff_summaries | chat: handoff cards, promise audits, routing, typed-Neuralese shift fan-in | 49 | 38 | w187, w201 |
| fact_chain_questions | reasoning seeded from StrategyQA **train** (facts as documents; records, not lone booleans) | 46 | 27 | w181, w196 |

- Most authored tasks are folder worlds with `expected_folders`, i.e. world-effect oracles. They are 30% of the
  final source's rows and about 90% of its authored rows.
- Constructs: decision readouts, skills, multi-operation worlds, iterate, and a few typed-Neuralese and relay programs
  (w180, w187, w201).
- The authors were asked to aim easier than earlier rounds (section 1). Their level mix in the final source is 103
  rows at level 1, 996 at level 2 and 256 at level 3. These totals include the store rows, whose levels come from the
  build.

**Round-8 rows (w163-w169).** 72 train rows were admitted on 2026-11-04, after the last pool build, so they never
reached `POOLS`. 65 of them enter now; custody left out the other 7 (section 5).

**Revived fallow source: the tau-bench store service.** `build-v3/{tau-retail,apigen-mt-retail,apigen-mt-airline}` was
built and verified on 2026-10-31 but never assembled into a pool.
- Each task gives the customer's own account of the request plus the retail or airline policy. The model must
  authenticate the customer, look things up and write through a `store` service over a fresh copy of the tau-bench
  database. The oracle is the store's final-state digest, which is tau-bench's own reward.
- Multi-step agentic tool use against a stateful service is otherwise absent from the pool.
- Selection (`round9/agentic/select-shard*.jsonl`): hash-ordered, at most one task per customer per source and at most
  460 per source.
  - 215 tau-retail tasks. These are tau-bench v1 **train** tasks; held tau users and all tau2 tasks were excluded at
    build time.
  - 331 APIGen-MT retail and 362 APIGen-MT airline tasks.
- Train users share no source group with the build's own dev/confirmation users (checked).

## 3. Qualification results

- **Infrastructure fix first (193a77d4b).** Since 60d2c1209, every authored qualification had been failing at custody.
  That commit removed `build_verified_teacher_trajectory_graphs` and `build_idiom_mixture`, and the held-set closure
  script imports both. The closure now runs from a frozen `git archive 60d2c1209^ scripts training` tree
  (`natlang-rl-authoring/protection-cache/legacy-src-60d2c1209p`). Its held-set inputs have not changed since.
- **Authored qualification.** `qualify_authored.py` ran against the grpo-v3 pin's runtime (`pin-7e929f2ea36c/ts-host`),
  not the frozen 2026-11-03 authoring runtime, in 2 GB ledger units, 1-3 cycles per batch. Every admitted batch passed
  every check:
  - the reference replay is accepted;
  - every wrong answer and every mutant is rejected;
  - every alias is accepted;
  - the smoke and replay smoke run;
  - custody and lint pass.
- **Novelty gate.** `authored_novelty.py` ran against the indexed full-core pool and `admitted/`. Every admitted batch
  passed, with 0 exact duplicates, 0 near duplicates and 0 functional repeats. Each first batch and each extension
  added 8-9 new programs. The gate's own functional counts are higher because world text inflates them, as the kit
  notes.
- **Blind solve.** An independent Sonnet solver worked each batch from the stripped packet alone
  (`round9/BLIND_SOLVER.md`), 754 tasks in all. A flagged task is held back even when its answer agreed. The 45
  held-back tasks:
  - 5 where the author's `expected` was wrong (w198: the labels counted distractors that the skill excludes);
  - 12 whose root text kept an unexpanded generator residue `''' + money_rule() + '''` (w185). They were rewritten,
    a 9th program was added, and the batch was re-admitted as b0002;
  - 3 whose program text contradicts itself (w176: "four required fields" lists three);
  - 3 whose declared `returns` type lacked a field that step 7 returns (w191 byline);
  - the rest flagged ambiguous by the solver: cue case, compound nouns, non-exhaustive lists and hedged dates in
    StrategyQA facts.
- **Store rows.** `verify.mjs` ran on the same pin runtime with the build's assets. All 908 rows were kept and all 908
  references accepted. All 2,694 wrong controls were rejected (other action sets, and doing nothing).

## 4. Repetition

`measure_repetition.py` against both current pools (`runs/natlang-rl-task-supply-20261117/repetition.json`):
- 1,355 rows with 213 distinct programs and 213 distinct instruction texts.
- 1 program text and 4 instruction texts also occur in the pool; these are siblings of the round-8 rows.
- 1,294 of the 1,355 rows are in families that are absent from the pool.
- Each of the three store families is one program: `serve_customer.nl` over different customers, orders and policies.
  That puts the source's effective program count at 26.8 and its top structure at 67%.
- The two parts are diverse in different ways. The authored part carries the program diversity: about 210 programs
  over 447 rows, with 2-3 worlds per program by design. The store part carries input diversity for one agentic program.

## 5. Held-data separation

- **held_eval verify.** Run against the held-v3 manifest (sha 45471060), with the source's 1,355 ids and its 2,019
  split/source groups as avoided: 0 overlapping ids and 0 shared groups. With the recorded training exposure
  (`ops/held-v3/exposure`) added: again 0 and 0 (`held-verify/`).
- **Exposure collector.** `ops/held-v3/exposure/collect_exposure.py` now lists the new pool, so later exposure
  collections resolve its groups. The original is kept as `collect_exposure.py.pre-supply-20261117`.
- **ProtectedWorlds.** The round-8/9 held rows were registered as `natlang-rl-authored-{dev,confirmation}-20261117r9`
  (133 + 117 rows) and the live registry was then checked.
  - The first build had 114 authored train rows related to held worlds. Every one was related through an
    `invented:<domain>` group string that the authors reused across sibling worlds, which the split then put on
    different sides. This is the known round-4 issue.
  - `build_supply.py --drop-related` leaves those rows out. The final source has 0 related rows.
- **LFM3 held-set group closure:** passed, 0 conflicts.
- **StrategyQA seeds** are train-split items only; `groups` carries their qids.
- `grpo.py`'s own exclusion (`held_eval.exclusion_ids` plus source-group siblings) still applies at load time.

## 6. Pool update for the next pin

**The source.** `grpo.py` `POOLS` gains `natlang-rl-task-supply-20261117/cases.jsonl`. It is registered in
`training/neuralese_corpora.json` with the manifest `training/corpus-manifests/natlang-rl-task-supply-20261117.json`.
- Any pin from this commit on uses it by default.
- To opt out, pass `--pool` with only the two earlier pools.
- Other readers of `grpo.POOLS` also pick it up:
  - `teacher_gen.py`: a new work dir draws its tasks from it; existing work dirs keep their frozen `tasks.jsonl`;
  - `exp1_measure_cases.py`;
  - `neuralese_variants.py`.

**Exposure is the binding constraint.** The new rows are 4% of the eligible pool (1,355 of 32,942). The walk over
unmeasured tasks visits the shuffled pool row by row, so a run sees few of them. Simulated over 20 seeds with the real
exclusions (`exposure.json`), for 600 explore groups:

| explore order | supply tasks reached | of them authored | supply families reached (of 24) |
|---|---:|---:|---:|
| `row` (current default) | 24 | 8 | 9 |
| `family` (new) | 44 | 37 | 22 |

- `--explore-order family` (6483ace06) walks the families round robin; the default stays `row`, so nothing changes
  unless a pin asks for it.
- It also spreads the rest of the pool's exploration evenly over the ~320 families, instead of following bgkit's row
  counts.
- Whether grpo-v4 uses it is the pin's declared choice, because it changes what is trained. Without it, this supply
  contributes a handful of groups per run.

**Base pass rates of the new tasks were not measured.** No engine was free: the DGX ran the run1 verdict and Pop ran
its evals. The run's explore share measures them, and a task found in the band then enters the learnable cycle.

## 7. Not checked

- **Pass rates.** No base or other model pass rate on the new tasks was measured. Difficulty rests on the authors'
  level targets and on earlier authored base rates. The store tasks may be near 0 for the 1.7B base, since tau-bench is
  hard; if their first explored groups all fail, cap or drop them at the next pin.
- **Blind solves.** Each batch had one Sonnet solver; no second solver was run. Doubts that an author reported but the
  solver did not flag were admitted.
- **Store task length.** Neither the store tasks' episode length against the 40-call budget nor their prompt length
  with the full policy text was checked.
- **Runtime JS gaps.** The authors hit these product-runtime gaps and worked around them. They were not investigated
  here:
  - `await` inside a `&&` or `!==` expression failed a smoke (w178);
  - `RegExp.test`, `match.index` and `matchAll` ended a root script (w197);
  - the identifier `transcript` ended a smoke before any call (w201);
  - a helper returning `record | null` was rejected (w199);
  - a root returning a field that its declared record type lacks was accepted (w191).
- **Custody losses.** Authored rows that reuse one `invented:<domain>` group across worlds still lose train rows to
  custody. Authors should use a group string per world.
- **Rotation ledger.** Round-9 batches were not recorded in it. A pending claim line that w184 wrote into
  `rotation.jsonl` expires after 6 hours.
