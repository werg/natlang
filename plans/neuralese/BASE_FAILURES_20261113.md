# Base LFM3 held-v2 failure taxonomy, and the harness/runtime/scorer fixes it led to (2026-11-13)

Data: the start policy (base 5ff07638) on held-v2 (256 tasks x 2 samples) on the current harness, run on the GRPO actor:
`/home/werg/data/lfm3-rl-dgx-20261029/ops/grpo-v1/run2/held/step-00000` (pin 37c221bfe, series e3a1066c7874). The outcomes were
229 task_success, 189 semantic_mismatch, 61 invalid_action, 31 budget_exhaustion and 2 unresolved.

Method. I read every one of the 283 non-successes and gave each one primary cause plus optional secondary causes. A
transcript digest tool printed each episode's invocations, tool calls and harness replies, plus expected vs actual
values and files. Budget stops were read from their partial logs. The classification rule was: a cause counts as ours
(harness, runtime, environment or scorer) only when the transcript shows the harness or the oracle decided the
outcome. Everything else counts as model capability. Per-episode records (path, primary, secondary, one-line reason,
excerpt, suggested fix) are in `/home/werg/data/lfm3-rl-dgx-20261029/ops/held-v2/base-failure-taxonomy-20261113.jsonl`
(283 lines). The counts below are from them.
Harness signals were also counted mechanically over all 512 episodes, successes included, from each episode's
`action_ledger` result texts.

## Taxonomy

| primary cause | episodes | whose job | outcomes (as scored) |
|---|---|---|---|
| **H-env**: task environment broken (repo/sql snapshot missing: ENOENT) | **45** (+2 unscored infra stops) | ours: fixed (e9a7360cc) | 37 invalid_action, 8 semantic_mismatch, 1 budget |
| **S-oracle-wrong**: the expected answer is wrong, or the task is ambiguous or garbled | **13** | ours (data): 1 generator bug fixed (3bb470c41), 7 upstream labels listed below | semantic_mismatch |
| **S-false-negative**: right answer, scorer said no | **7** (3 clear, 4 borderline) | ours (scorer policy): documented, not changed | semantic_mismatch, 1 invalid_action |
| H-tool-surface / H-error-misleading / H-unrecoverable (primary) | 8 | ours: fixed where general (below) | mostly budget_exhaustion |
| M-procedure: required file effects skipped or botched (moves not made, `cp` for `mv`, wrong folder, missing literal words in a line format) | 71 | training | semantic_mismatch |
| M-reason: wrong judgment, reading or arithmetic with a workable process | 54 | training | semantic_mismatch |
| M-loop: budget burned on repeated invalid or redundant actions the errors did address (nested `nl` self-delegation, read_code on invented names, `nl(...)` called as a function) | 24 | training | budget_exhaustion |
| M-format: right content, wrong shape (extra keys, path prefixes, full labels for names) | 22 | training | semantic_mismatch |
| M-giveup: returned blocked/failed on a doable task | 21 | training | invalid_action |
| M-fabricate: returned values never computed from evidence | 16 | training | semantic_mismatch |

In all, 208 of the 283 failures (73%) are model capability and 75 (26.5%) are ours: 47 environment, 20 scorer/oracle and
8 harness as the primary cause. Harness friction was a secondary cause in 44 more model failures (H-tool-surface 27,
H-error-misleading 9, H-reject-idiom 8).

Held-v2 therefore understates the base. 45 episodes could not succeed at all, and 8 tasks (13 failed episodes) have a
wrong or ambiguous oracle. Without those and the 2 unscored stops, the base is at 229/452 = 50.7%, against 229/510 =
44.9% as scored.

### Harness signals over all 512 episodes

Episodes in which each harness message occurs, and its share among failed (283) vs successful (229) episodes:

| signal (harness text) | episodes | in failed | in successful | status |
|---|---|---|---|---|
| `read_code: no-such-function` (own call name, invented helpers, file paths, type names) | 272 | 60% | 44% | message already explains; model capability |
| `X is this call's input and already holds the caller's value` (input re-typed as a literal) | 160 | 24% | 40% | not discriminative; leave |
| `transcript.search/entry: bad-action, expected a scope-eval tool` | 65 | 18% | 6% | **fixed** (45f7c0ddb) |
| `read_page: No tool output was cut off` / `no-such-page` | 58 | 16% | 6% | message already names the eval form |
| `return: type-mismatch` | 58 | 12% | 11% | JSON-text results fixed earlier (802e111bc) |
| `bash cannot bypass the bash policy` (`bash -c '...'`) | 50 | 15% | 3% | **fixed** (be6194095) |
| `ENOENT ... assets/repos/*.json` | 45 | 16% | 0% | **fixed** (e9a7360cc) |
| `is already a function / a built-in of this scope` | 47 / 36 | 10% / 9% | 8% / 4% | messages are fine |
| Python written into eval | 22 | 5% | 3% | hint **corrected** (45f7c0ddb) |
| `services.neuralese is missing or has no read body` after `read(value)` | 17 | 4% | 3% | **fixed** (45f7c0ddb) |
| `mv: target 'dir/' is not a directory` after `mkdir -p dir` in an earlier bash call | 9 | 8 eps | 1 ep | **fixed** (be6194095) |

## Ours: fixes landed on main (for the next experiment boundary; no running pin was touched)

1. **The codebase-task assets were missing on the DGX: 45 failed episodes, 27 tasks, 54 episodes now scorable (e9a7360cc).**
   Every `ext_codebase-{natlang,py}_codebase_*` service reads `repos/*.json` from
   `NATLANG_RL_TASK_ASSETS_V3 ?? /home/werg/natlang/runs/natlang-rl-broad-external-20261031-v3/assets`, and that directory
   never existed on the DGX. The same files are in `/home/werg/data/broad-rl-tasks-20261031/assets-v3`. All 23 repo
   snapshots that held-v2 uses were checked there against the sha256 pinned in each service. `run_bonsai_queue.run_queue`,
   which all collection (held_eval, GRPO, exp1) goes through, now sets the variable to the build tree when it is unset
   and that tree exists. Each service still checks its file's sha256. EPISODE_COST_20261112 had found this for training
   (45 episodes, 0/47) and GRPO excluded the families, but held-v2 kept scoring them as policy failures.
   - Consequence: base results on held-v2 from before this change are not comparable with later ones. Re-measure the base
     on the next pinned commit.
   - The GRPO exclusion `^ext_codebase-(py|natlang)_` can be dropped at the next experiment.
   - The two SQL tasks and one databench task read the v2 assets, which exist. Their failures are the model's.
2. **`bash -c '...'` runs, and directories made with `mkdir` persist across bash calls (be6194095).**
   - The base wrapped commands in `bash -c '...'` in 50 episodes (96 refusals). In 42 of those episodes it then failed;
     in row 54/0, 54/1 and 137/1 it never made the required moves after the refusal. The inner literal script now
     goes through the same loop/recursion policy and runs. A non-literal script, `eval` and `source` stay refused, with a
     message that names the literal form.
   - An empty directory made by `mkdir -p filed` was forgotten when that bash call ended, so `mv x filed/` in the next
     call failed with `target 'filed/' is not a directory` (9 episodes, 8 of them failed). Empty directories are now
     kept per Folder for the whole call.
3. **A tool call that names something in the eval scope gets the eval form (45f7c0ddb).** `transcript.search`,
   `transcript.entry`, `transcript`, `read_function` and other names sent as tool calls (80 actions in 46 episodes, 33 of
   them failed) used to get only `expected a scope-eval tool`. They now get, for example: `transcript.search is not a
   tool: transcript is in this call's eval scope. Use it in code, for example eval with code: transcript.search("freight");
   read_code("transcript") shows how to use it.`
4. **`read(value)` works without a Neuralese standard library (45f7c0ddb).** Our own opaque-access error says "Read it
   with read(value)", but held collection configures no `services.neuralese`. So `read()` failed with "Converting a
   Neuralese value to text needs typed readout: services.neuralese is missing..." (17 episodes, 11 failed). It was the
   primary cause of row 186/1, 232/1 and 252/1, and a secondary cause of row 186/0, 195/0, 145/0 and 144/1. `read()` is
   now a typed `natlang.read` call of the executing model that takes the value as its argument, as the teacher-text
   readout already was.
5. **Python-in-eval hint (45f7c0ddb).** The hint told the model to "write judgment as `await nl(...)`", and the
   `nl-not-tag` check then refuses that (row 7/0, 6/1). It now shows `await nl\`instructions\`(value)`.
6. **`decide()` record compared with an answer (c0b7ac776).** In 172 places, authored task text binds
   `const verb = await decide(...)` and then branches on `verb` as a string. The families are auth_request_routing_gates
   (31), auth_ops_directory_reducers (28), auth_exact_algorithm_state_workflows (24), auth_harvest_ledger_operations
   (16), auth_fact_desk_escalation (15) and others. The base followed held row 91/0 literally, no branch ever matched,
   and it returned the input unchanged. Eval now refuses such a comparison before it runs: `verb is the decision record
   that decide(...) returns, not its answer: compare verb.value.` The task texts themselves should be corrected when the
   pool is next rebuilt.
7. **Wrong oracle from a generator bug, which corrupts training rewards (3bb470c41).** `natcode._type_aliases` dropped a
   type's last member when it was written just before `}` with no `;`. Held row 246 (`HistoricPlan`, both samples): the
   base answered all six fields and was scored wrong because the oracle left out `evidenceRef`. **In the RL pool
   (b3fef4c7), 377 of 432 `explore-fields` rows and 3 of 8 `explore-optional-fields` rows have this wrong oracle.** A
   correct answer scores 0 on them, and only an answer that omits the last field scores 1. Their measured base pass rate
   is 0 (exp1 `base-pass-rates.json`, 4 rows), so GRPO's 85% band never draws them. The 15% unmeasured share can draw
   them, though none of run2's first 122 groups did. The corrected expectations are in
   `/home/werg/data/lfm3-rl-dgx-20261029/ops/pool-errata/natcode-explore-fields-last-member-20261113.jsonl` (380 rows:
   id, type, expected_was, expected). The next experiment's pool must apply this errata or exclude these ids.
8. The serving front now logs the traceback of any request it answers with 500 internal (246d83979). Row 110/0 stopped on
   `model HTTP 500: IndexError: list index out of range` after a 45k-character reasoning turn, and no log showed where.
   That episode is `unresolved` (unscored), so it did not affect the score.
9. A parse error no longer also produces `Saved helper "main" cannot persist because it refers to unavailable ;` (an
   empty name; 6 episodes).

Tests: `ts-host/test/folder-shell.test.mjs`, `compaction.test.mjs`, `scope-compiler.test.mjs`,
`neuralese-read-contract.test.mjs`, `eval-field-check.test.mjs`, `tests/test_bonsai_queue.py`,
`tests/rl_tasks/test_natcode_missing_child.py`. The model-facing ones are recorded in `plans/MODEL_FACING_CHANGES.md`.

## Ours: scorer false negatives and held-v2 oracle errata

Update 2026-11-13: held-v3 replaces these 8 tasks and fixes the 3 clear false negatives plus row 30/1 in the scorer, from the next boundary (`HELD_V3_20261113.md`). Rows 223 and 189 stay as they are.

Scorer false negatives. They are correctness issues and small in count: 3 clear episodes, 4 borderline.
- **Span QA with a one-word gold (3 episodes).** The span oracle is token F1 >= 0.8 after SQuAD normalisation, so any
  extra word fails a one-token gold.
  - Row 23/1: "What was attached to the base of the water tower?" The model answered "the alien's spaceship", gold "The
    spaceship" (F1 0.5).
  - Row 52/0 and 52/1: "What did Co-workers A say they just had...?" The model answered "just had sex", gold "Sex" (F1 0.5).
  - The principled fix for 52 is to drop answer tokens that repeat the question before computing F1. The checker does
    not receive the question today (`checkOracle(actual, expected, spec)`), and the oracle spec is part of the pinned
    pool data. A containment rule ("gold inside a short answer") would also accept row 66's "Duke Lemur Center" for
    "Duke" (the question asks for the university), so I did not add one.
  - Changing span scoring changes the held series. Decide it together with the next held-set revision.
- Row 53/0: `rejected.md` had exactly the right line but no final newline. The task never asks for one, and the
  `content` compare is byte-exact. Some sibling tasks do require the newline ("End seats.md with a newline"), so the fix
  belongs in the task data (`markdown-terminal-newline` where the text does not say), not in the default compare.
- Row 30/1: the model ran the real `resolve` step in a bounded `for` loop, saw the conflict count cycle at 1 and returned
  `failed` for the right reason. `stop_evidence` requires the iterateOn step-trace hash
  (`"phase":"step","step":7,"state_hash":"4aba14649352c7f1"`), which a plain loop never emits, and the task text does not
  require iterateOn. Fix in data: an alternative decisive marker, or task text that names iterateOn.
- Rows 223/0 and 223/1 (qa-alias-map): "secretly in private churches" against gold "as long as their services took place
  secretly in private churches". This is a partial span (F1 0.53). I count it as borderline and leave the scoring as it is.

Held-v2 tasks whose oracle is wrong or whose task is ambiguous (8 tasks, 13 failed episodes). Exclude them from comparisons or fix them in
the next held set:

| row | task id | problem |
|---|---|---|
| 142 | `ext:multihiertt:multihiertt-arithmetic-3tables:0815bb8c...:c7636aae` | expected 79 = 47 + 32 double-counts the 32 semisubmersibles that are part of the stated 47 rigs; model 47 |
| 239 | `ext:multihiertt:multihiertt-arithmetic-3tables:17b11573...:a6b81112` | page gives $1.0/1.2/1.7 bn, average 1.3 (model); expected 0.43137 |
| 246 | `ext:natlang-codebase:...-fields-historicpla:d92d67ff` | generator bug above (evidenceRef dropped) |
| 193 | `ext:decisions:route-in-k4:route-massive-intent-train-80:78d5bc42` | "recipe for chicken thighs" labelled cooking_query, not cooking_recipe (both samples chose recipe) |
| 216 | `ext:databench:databench-sample-category:007-fifa-8-sample:26246fcd` | five-way tie for the most common nationality; the oracle takes the first |
| 251 | `ext:triviaqa:triviaqa-1pages-1:dpql-1855:6c29c6a3` | the one page given (Antarctica) never mentions an airport |
| 225 | `ext:mbpp:mbpp-3tests:620:1cca5ab9` | text asks for "the largest subset"; the tests expect its size |
| 244 | `ext:mbpp:mbpp-3tests:628:fc976a53` | upstream text garbled ("with character * list item * list item ... '%20'") |

## Ours, open (smaller; for the owners of those areas)

- **The opening shows the directory-reducer paragraph to every call with code of its own** ("Call await
  someFolder.apply(reducer, ...)", `FUNCTION_TOOLS_PROMPT`), including calls with no folder and no reducer. It invites
  `read_code("someFolder")` (9 calls), `.apply`/`.propose` lookups, and giving up (rows 127/0, 127/1, 167/0). Show it
  only when the codebase has a directory reducer. This touches `callNotes` and the training rebuild in
  serving-request.ts (Roo Code's area), so it was raised by coord note.
- `read_file` appends `(in eval: const text = await folder.file("...").readText())` after the content. In 2 failed
  episodes the model copied it into files it wrote (row 54/0 into nine `retired/*.md`, row 109/1 into `pytest.ini`).
- In a consumer call, a Neuralese argument shows as `{$neuralese:{id, source_occurrence}}`. The occurrence id looks like
  a `read_page` id, and consumers spend 20+ calls on `read_page`/`read_code nzs1_...`/`transcript.search` (rows 117/0,
  117/1, 92/1, 144/1, 195/0). `read(n)` now works. The listing could say so on the argument line.
- The loop checker treats a nested `for (let i ...)` that shadows the outer `i` as assigning the outer counter (row 20/0,
  1 occurrence).
- In the GRPO run, 97 training episodes are `infrastructure_error` with 0 model calls, and 113 are `unresolved`, out of
  904 in 122 groups. They are unscored (reward null), so they do not corrupt rewards, but they are lost compute around
  restarts. This is for the GRPO operator.

## Model capability (training's job)

The largest model class, about a quarter of all failures, is **the right answer with the required effects missing**.
The model computes the counts correctly and returns them, but the files are never moved, are copied instead of moved,
or are moved into a different folder (64 of the 71 M-procedure episodes). Next come wrong judgments made by eye when
the task names an `nl`/`decide` helper (M-reason, 54), giving up with `blocked` on doable tasks, often right after a
recoverable harness error (21), and loops of self-delegating `nl` children or `read_code` on invented names until the
64-call budget runs out (24). These are what RFT/GRPO on effect-state tasks should move, and the per-call sequences
already carry them.
