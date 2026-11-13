# GRPO task supply: the learnable band and Neuralese-declaring variants (engineering, 2026-11-13)

Question: GRPO (`scripts/neuralese_data/grpo.py`) draws 85% of its groups from tasks whose measured base pass rate lies in
[0.1, 0.9] and 15% from unmeasured pool tasks. Is the band too small, can family/level information find more learnable
tasks, and what goes into the first Neuralese addition (Neuralese-declaring variants, as data)?

Data: `ops/exp1-rft/base-pass-rates.json` (768 tasks; `pass-rate-cache.json` holds the same 2,362 samples per episode),
run2's own groups (`ops/grpo-v1/run2/groups/*/rewards.jsonl`, 91 scored groups on 73 tasks), the GRPO pool (33,164 rows,
28,231 eligible after the held-v2 closure and the codebase-repo regex; 301 families). Scripts of this analysis were
scratch; the numbers below are reproducible from those files.

## 1. The band

- Measured with >= 2 scored samples: 697 tasks (681 eligible). Rate histogram: 212 at 0, 334 at 1, 151 in [0.1, 0.9]
  (150 eligible). Most tasks have only 2-3 samples (312 have 2, 218 have 3). Run2's live count of learnable tasks (base
  rates overridden by its own latest group) was 130-139.
- 298 of 301 families are measured, but by 1-3 tasks each; the measurement covers families widely and tasks thinly.
- Within-family heterogeneity dominates. A beta-binomial fit over measured tasks (family mean shrunk to the global mean)
  gives a concentration kappa ~ 0.5: task success rates inside one family are U-shaped (mostly near 0 or near 1),
  whatever the family mean. Under that fit an unmeasured task's group of 8 has mixed outcomes with probability ~0.36-0.44;
  a measured in-band task's (posterior) ~0.81, observed 0.78 (61 of 78 run2 groups on base in-band tasks had spread;
  explore groups 5 of 8).
- By source (tasks with >= 3 samples, share with mixed outcomes): bgkit 6/12, codebase-ops 14/29, natlang-codebase 12/26,
  decisions 13/30, auth 22/71, design-brief-gates 13/51, curriculum 6/34, reasoning-gym 6/40, babi 2/16. In-sample this
  looks like a usable prior (bgkit, decisions and natlang-codebase high; wikisql, math, hotpot/2wiki saturated).

**Family priors do not predict out of sample.** Half/half task splits (30 seeds), predicting whether a held-out task's
own 2-3 samples are mixed: global prior log-loss 0.510 / AUC 0.623 (its only signal is the sample count), source-group
prior 0.527 / 0.608, family prior 0.519-0.554 / 0.587-0.614 depending on shrinkage. Family and source means carry no
information about which individual task is mid-range beyond noise, so weighting exploration by family (in-sample it
looked like 0.44 -> 0.59-0.75 spread) would be fitting noise. Not implemented.

**Level adjustment is not a lever.** Within families with >= 2 measured levels, the highest level is lower than the
lowest in 18 families, higher in 18, within 0.1 in 19; pooled by absolute level 0.63/0.69/0.49/0.57 (levels 0-3);
relative to the family's level range low/mid/high 0.60/0.41/0.60. `rl/curriculum`'s level shifting cannot move tasks
into the band on this pool; 179 of 301 families have a single level anyway.

**What the data does support: more exploration.** On run2's own groups, a visit to a learnable task ends 0/8 or 8/8 in
22% of groups (17 of 78), which removes the task from the band (latest own group replaces its rate), while an explored
task enters it with probability ~0.4. With explore share e the learnable set changes per group by
`e * 0.4 - (1 - e) * 0.22`: at e = 0.15 that is -0.13 tasks per group, so the ~139-task set drains in roughly
1,000 groups (50-90 steps at 12-24 groups per step) and each remaining task is revisited every ~8 steps meanwhile. The
set is stable at e ~ 0.35. Cost: the useful-group (spread) fraction falls from 0.85*0.78 + 0.15*0.4 = 0.72 to
0.6*0.78 + 0.4*0.4 = 0.63 at e = 0.4, in exchange for a learnable set that grows (+0.03 tasks/group) and a less
repetitive supply (2.7x as many new tasks per step).

**Change (main, for the next restart): `--explore` default 0.15 -> 0.4.** Running pins keep their own copy (run2 runs
0.15). Exploration remains a uniform walk over unmeasured eligible tasks; a measured out-of-band task is never
re-explored (posterior spread 0.22, below an unmeasured task's).

## 2. Neuralese-declaring variants (the first Neuralese addition, as data)

Finding first: the pool is already about half Neuralese-declaring. 14,817 of 28,231 eligible tasks declare
`Neuralese<...>` (composed neuralese_{iterate,typed,fanin,relay}, multihop_deep, bgkit readers, decisions_verdict, half
of the natlang-codebase/SQL/multidoc units through `make_case`'s per-unit `neuralese-helper` framing). Their base rate is
similar (0.56 vs 0.60), but only 17 of 71 measured ones are in band (vs 133 of 610). GRPO run2 therefore already trains on
Neuralese-declaring programs; what was missing is the paired form (the same task with and without the declaration),
which is what a controlled addition needs.

Batch `natlang-rl-neuralese-variants-20261113` (`scripts/rl_tasks/neuralese_variants.py build`): the `neuralese-helper`
framing of existing plain tasks. A variant adds one callable folder helper returning `Neuralese<string>` (a private
per-document/per-item/per-part note) and one root-note paragraph; inputs, expected value, oracle, wrong/alias checks,
reference, split, `split_group` and `source_groups` are the original's (so the held-v2 closure that excludes an original
excludes its variant), and its id is the one `make_case` gives the neuralese framing of that item.

- Generator helpers (12 families, 1,284 rows with the authored families below): helper file and root note read from the
  pool's own neuralese-framed rows of the family (natlang-codebase read_function, musique read_document, bird/spider
  plan_query, quality read_part, hybridqa read_page, multihiertt read_table), so the variant is byte-for-byte what the
  generator emits for that unit.
- Family helpers written in the script (`AUTHORED`, 14 families): babi note_facts, clutrr relation_in, logiqa2
  weigh_option, drop read_part, tatqa read_paragraph, finqa/convfinqa read_text, wikitablequestions read_rows.
- Per-task helpers for measured in-band tasks whose family has neither (118 tasks; 6 Haiku subagents, one helper per
  task with the task's own program and inputs in view): 97 usable (design-brief gates note_evidence_page/note_decision,
  auth desks note_claim/note_invoice/..., decisions read_ticket, codebase-ops read_module/read_move, natlang-codebase
  repairs note_project_file, ...), 21 judged not applicable (single short puzzle strings: reasoning-gym, math, mbpp,
  toolace, one route request, three curriculum tasks). `task_helper_problems` checks name, arity, argument types, the
  note request and the root note before replay.
- Selection: eligible, plain-framed, non-declaring rows of families whose measured base rate is in [0.1, 0.9], plus every
  measured in-band task; at most 120 per family. 1,381 rows, 47 families; 112 of the ~155 measured mid-range tasks have
  a variant (they are what GRPO mostly picks).

Qualification (`scripts/rl_tasks/verify.mjs`, the collector's execution path; reference replay must be accepted, every
wrong answer rejected, aliases measured), run against a frozen copy of `ts-host/dist` at 3bb470c41 (executable sha
b0f9f954; the live checkout's dist was rebuilt mid-run once and the verifier refused it): **1,381 of 1,381 kept**, all
references accepted, 3,866 wrong-answer controls all rejected (0 errors), 1,378 alias probes all accepted; the 97
Haiku-authored helpers all load and replay. A control run on deliberately broken helpers (unknown type, malformed
frontmatter) rejected both at load. Held-v2 closure re-checked on the pool plus the batch: no variant and no original in
it, none shares a held source group, no id collides with the pool. Repetition
(`measure_repetition.py` against both pools): 451 distinct instruction texts (effective 107), 490 programs (effective
113), 31 structures; 202 rows have a normalized program text already in the pool (the generator's neuralese framing of
other items of the same template), no family is new to the pool. Variants are not padding: each is a distinct framing of
a distinct item, and an item and its variant enter as a pair only behind the flag.

Registered as `natlang-rl-neuralese-variants-20261113` (`training/neuralese_corpora.json`, manifest in
`training/corpus-manifests/`). Use: `grpo.py train --neuralese-variants runs/natlang-rl-neuralese-variants-20261113/cases.jsonl` (off by default;
`--neuralese-variant-share 0.5` runs the variant in half the groups of a task that has one, and a variant's outcomes are
its own, never its original's band rate; `WORK/neuralese-variants.json` records path, sha and counts). It enters as its
own controlled addition after two consecutive held gains from plain updates, with a paired held-v2 measurement.
