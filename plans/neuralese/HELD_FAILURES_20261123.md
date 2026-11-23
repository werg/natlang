# Why the current policy fails held tasks (2026-11-23)

Offline analysis of three Pop held measurements (2 samples per task, no GPU):

| Set | Policy | Held set | Episodes | Passed |
|---|---|---|---|---|
| v7 | sft-v7 (reference, `ops/ei/sft-v9/held/start/`) | held-v3 | 512 | 308 |
| v9 | sft-v9 candidate e15b20a8 (`ops/ei/sft-v9/held/candidate/`) | held-v3 | 512 | 311 |
| v6h4 | sft-v6 (`ops/held-v4/runs/sft-v6-pop/`, run in progress) | held-v4 | 329 finished | 130 |

Every failed episode was digested from its result and trace (or, for budget stops, its partial evidence). Seven
independently briefed readers labelled each one from its trajectory: the surface outcome, the deeper cause, the construct
involved, the turn, the evidence and a remedy. 582 episodes were labelled: v7 204, v9 201, v6h4 177. The 22 unfinished
held-v4 episodes were skipped.

The labels, digests and scripts are in `/home/werg/data/lfm3-rl-dgx-20261029/ops/ei/sft-v9/held/failure-taxonomy-20261123/`:
- `labels_all.json`, `flips_all.json`
- `episodes.jsonl` (one row per episode)
- `children.json` (child calls and Neuralese reads per episode)
- `coverage.json`, `aei1_prefix.json`
- `digests/`
- `extract.py`, `readback.py`, `children.py`

Those files quote held inputs. Never use them as training seeds.

## 1. Surface outcomes

| Surface | v7 | v9 | v6h4 |
|---|---|---|---|
| wrong answer | 86 | 80 | 71 |
| 40-request budget stop | 50 | 48 | 94 |
| wrong files | 34 | 41 | 3 |
| wrong answer and files | 24 | 15 | 8 |
| wrong status (should stop, refuse or report failure, or the reverse) | 6 | 12 | 1 |
| hinted technique not used | 3 | 5 | 0 |
| infrastructure | 1 (an abort at 2656 s) | 0 | 0 |

Timeouts were 2 in v7 and 0 in v9. There were no malformed-call failures as such: every malformed call was recovered
from or ended in one of the rows above.

## 2. Deep causes

| Cause | v7 | v9 | v6h4 |
|---|---|---|---|
| misread_spec: a stated rule, destination or edge case skipped | 33 | 43 | 29 |
| tool_idiom: wrong eval API, bash file mechanics, re-running child loops | 31 | 25 | 33 |
| harness_or_task_suspect (mostly Neuralese, see 3.1) | 17 | 23 | 47 |
| bypassed_construct: did by hand what a named function, `decide`, `nl` or a service should do | 18 | 20 | 29 |
| exploration_loop | 16 | 15 | 15 |
| output_format: a literal word, newline or prefix dropped | 16 | 14 | 2 |
| other: a child decision readout chose the wrong label while the root was correct | 16 | 14 | 2 |
| error_recovery | 12 | 10 | 14 |
| premature_stop | 7 | 12 | 1 |
| incomplete_coverage / over_inclusion (codebase exploration) | 11 / 7 | 4 / 8 | 0 / 1 |
| compute_error / parse_error | 10 / 7 | 5 / 6 | 1 / 0 |

Readers' remedies, v7+v9: training_data 315, runtime_error_message 37, harness_bug 30, task_bug 17. For v6h4:
training_data 101, harness_bug 45, runtime_error_message 28.

The construct at the heart of each failure, from keyword buckets over the readers' `construct` field:

| Construct | v7 | v9 | v6h4 |
|---|---|---|---|
| file effects (move, write, folder API, bash) | 49 | 36 | 14 |
| child natlang calls and `nl` lambdas | 41 | 42 | 75 |
| decision readouts (`decide`, label children) | 32 | 29 | 57 |
| Neuralese read and consume | 28 | 30 | 14 |
| code (python, mbpp, TypeScript) | 13 | 14 | 1 |
| codebase exploration (`read_code`, search) | 12 | 11 | 0 |

## 3. Findings, by size

### 3.1 Neuralese values cannot be read back: a harness or representation defect, not a skill gap

`natlang.read` (`natlang:model-read/1`) is the template-readout child that decodes a Neuralese value. `readback.py`
compares its output with the text the producer returned for the same `nzs1` state. It matched in:
- **1 of 87** read-backs in v7;
- **1 of 87** in v9;
- **0 of 24** in v6h4;
- **13 of 465** in aei1's own DGX collection.

The other reads return unrelated text, for example:
- "Register after 0 notices.\nApplication PL/26/0519." read back as `jinja2`;
- a landing record read back as `{ "users": [...] }`;
- an object read back with the right keys and empty values: `{answer: "", sentence: ""}`, `{totals: []}`.

The requests do carry the state: `native_state.imports` is `exact-native-state` with the producer's id. So the reader
decodes with the state attached and still sees nothing usable. That holds on Pop SGLang (BF16, vendor projection, FP8
KV) and on the DGX actor alike.

Consequences in the labels:
- 28 / 30 / 14 failures (v7 / v9 / v6h4) are Neuralese consumers or readers that block ("the note is not available")
  or loop to the budget.
- Programs whose spec routes data through Neuralese children fail when the model follows the spec. Among tasks with
  callables (`children.json`, v7+v9), episodes pass as follows:

  | Episodes | Passed |
  |---|---|
  | no child calls at all | 122 of 190 (64%) |
  | child calls, no Neuralese read | 180 of 330 (55%) |
  | at least one Neuralese read | 23 of 69 (33%) |

- Several flips are exactly this. In `section-carry` migration-tickets (2 → 0), v7 passed by computing in one eval and
  v9 followed the Neuralese chain. In costume-settle, referral-queue and correspondence-games, the spec path failed and
  the bypass path passed.

Expert iteration therefore trains on successes that bypass the program's callables, and the reward currently favours
not following the program. This is the first thing to fix. A text-primed state of a two-line string must read back
exactly. AGENTS.md: "A span that ablation cannot tell from shuffled state is a defect."

**Not found:** where the content is lost. Candidates:
- the producer's state is not the text-primed span;
- the import binds the wrong positions;
- the template readout ignores it;
- the reader is untrained.

Assigned as a bug hunt in `plans/semantic-fp/OPEN_ITEMS.md`.

### 3.2 The 40-request budget charged a decision readout 8 times (fixed)

A `readout: decision` call sends 4 reasoning samples and 4 option scorings, and the collector counted all 8. In
`sfp-untrusted-contract` row 087, six `urgency` readouts used 37 of the 40 requests while the root used 3. In held-v4,
`sfp-decision-calibrated` passed 1 of 15. That accounts for most of v6h4's 94 budget stops, and for every v7/v9
decision program with five or more decisions or a per-file child chain (fish-landings: 14 files × 3 children).

**Fix:** a reasoned decision now counts once (`teacher/collector.ts`, opt-out `NATLANG_BUDGET_EACH_DECISION_REQUEST=1`).
This starts a new held scoring series, so the base is re-measured on it.

### 3.3 Child decision readouts choose wrong labels while the root is right

v7 16, v9 14, plus 24 more in the decision cross-tab. Examples:
- near-miss requests that should be out_of_scope (banking77, clinc);
- fine TREC labels;
- spam versus ham with money words;
- routing a "charged for a dropped call, credit it back" note to network instead of billing.

This is a calibration gap of the decision readout itself.

**Remedy:** decision-readout training data on the label sets' hard negatives (task brief D in OPEN_ITEMS), through
expert iteration's own pool. The held-v4 readers also found that in three 2–2 ties of the four samples, the tie went to
needs-review each time; the tie-break rule is listed for checking.

### 3.4 Spec misreads in directory reducers and file effects (the largest training-data class)

v7 33, v9 43. Patterns:
- destination folders created under the input folder instead of at the root (`claims/paid/` for `paid/`);
- copying where the spec says move;
- writing the summary but skipping the moves (finishing a reducer as a report);
- a "stop when" cut-off ignored;
- fees written for unallocated applicants (3 of 3 episodes);
- a null field treated as a value;
- name versus path in output lines;
- literal template words dropped (in ferry-pass-close, all 4 episodes dropped "crossings", "fare" and "carried").

Bash file mechanics add to it:
- `mv "$f" dir/$f` doubles the path;
- `rm -rf` cleanups destroy work;
- the sandbox `mv` refuses to overwrite with "cannot safely determine whether … are the same file" (12 digests).

**Remedy:** task brief A, plus the harness items in 3.8.

### 3.5 Children behave like fresh top-level agents; child loops re-run

Observed patterns:
- a child tries to call itself ("start_notes is the call you are executing"; 11–12 occurrences per name);
- a child hands its question to a new `nl` call with the same instructions ("quiesced: not run: it has the same
  instructions and inputs", 16);
- the root re-runs a whole loop of child calls in the next eval instead of reusing the stored results, which burns the
  budget (v7/v9 rows 045, 080, 154, 009, 142; snow-calls; hollowmere);
- structural recursion on the same input.

The runtime messages already teach the fix, and the model still does it.

**Remedy:** training data (task brief B) and teacher demonstrations for deep child chains (row 010: a 5-level chain
in which every child acted as a fresh agent).

### 3.6 Codebase exploration: coverage and precision

`ext:natlang-codebase` held is 20 of 40 in v9 (pool 1769 tasks, historic rate 52%). The patterns:
- the children task lists the parent itself;
- the mentions task matches loosely: frontmatter args, or "impact" for "impacts";
- the leaves task misses leaves nested in sub-namespaces;
- the root operation or the root prefix is omitted.

Passing episodes read 12–17 functions and failing ones read 3–4: pass or fail follows read coverage. The `read_function`
helper got "expects 3 arguments, got 1" 18 times; that message is fixed (3.8).

**Remedy:** task brief C (exhaustive-coverage exploration with exact path format).

### 3.7 Families at zero: no own successes to imitate

47 held-v3 tasks failed in all 4 episodes. The authored families among them have near-zero historic pool rates:

| Family | Historic pool rate |
|---|---|
| section-carry-obligation-readers | 0/27 |
| request-to-call-queues | 0/8 |
| damaged-folder-reconciliation | 0/17 |
| edit_stream | 0/8 |
| procedure-library-selection | 6/98 |
| fact-desk-escalation | 7/84 |
| quantity-reasoning-desks | 5/30 |
| multi-round-state-games | 10/65 |

aei1's collection sampled almost none of them (0–2 episodes each) and mostly collected ext families (design-brief-gates
10% of episodes, held 1.2%; reasoning-gym 6%, held 0%). Authored families are 39% of held-v3 but 28% of aei1's
episodes. Own best-of-k cannot produce targets at a 0% rate. Several of these families are also the Neuralese-chain
families of 3.1, so the fix of 3.1 comes first.

**Remedy:** teacher tier (`jolt-guided` or teacher trajectories with student reasoning) for these families after 3.1,
and task brief E for fresh siblings of these capabilities.

### 3.8 Runtime messages and harness behaviour

Fixed in this change (`plans/MODEL_FACING_CHANGES.md`, 2026-11-23 entry):
- `await` inside a non-async arrow (`files.map(f => … await f.readText())`) now fails at compile with the fix named.
  Before, it surfaced as "SyntaxError: Unexpected identifier '__natlang_own'", an internal name. That happened 23 times
  in 13 failed episodes and never in a pass.
- An arity error now lists the parameters: "read_function expects 3 arguments (question: string, name: string, source:
  string), got 1".
- A `for … of` over an unawaited Promise now says to await it. It used to say "plain object; iterate Object.keys".
- `refine()` in eval failed under every decision-readout collector with "decision readout needs its invocation
  identity", because the judge had no invocation id (v6h4 rows 119, 121, 128).
- The whole-case budget counts a reasoned decision once (3.2).

Found but not changed (OPEN_ITEMS, owner named there):
- After a `console.log` that printed the full value, an eval still shows "Stored local doc = … <<cut off: 755 of 835
  characters not shown>>". This drove hand transcription and `finish:true` calls made just to look at the document.
- A rejected service write reads like a partial write ("Already performed before the failure (not undone):
  board.commit_move").
- `finish:true` accepts record fields outside the declared return type, and the exact oracle then rejects them
  (v9 rows 077, 062, catalog15, permit55).
- `return_result` refuses a JSON string that parses to the declared record, and its hint adds an unrelated `nl.with<T>`
  paragraph. The same paragraph appears on other type errors ("expected a portable value").
- Bad paths are taken silently:
  - `moveTo(folder.dir('x') + name)` creates `[object Object]name`;
  - `moveTo(folder.dir('traced/s01.md'))` nests the file under itself;
  - `folder.file(fileObject)` gives "file not found: adjustments/[object Object]".
- The sandbox `mv` will not overwrite an existing file, with a misleading message.
- `read_code("refine")` answers "no-such-function" for a built-in.
- "Binding "x" is immutable in eval" never says the variable was declared `const`, and models concluded that
  top-level variables are banned.
- Callable shadowing is caught only partly: `const intent = await intent(x)` gives a raw temporal-dead-zone (TDZ)
  ReferenceError.
- Leaked `</think><tool_call>` text inside eval code produces dozens of TypeScript errors instead of one sentence.

### 3.9 Task and oracle suspects (task_bug, 17)

| Task | Problem |
|---|---|
| row 130 (codebase-ts-effects) | The gold names a function with no network effect in its body. |
| row 095 (finqa composed) | The gold uses a different line item than the question names. |
| row 235 (bAbI path) | The letter format is never stated. |
| row 249 (ToolACE) | The gold expects `module_id: "A"` for "Module A". |
| row 204 (databench) | Two height columns disagree. |
| row 143 | The passes look spurious: the evidence chain broke and "no" happened to be the label. |
| QA span oracle | Scored "I just had sex." 0.4 against "Sex". |

These go to the held-v3 errata review: 6 held tasks with a suspect gold. Owner: next held errata pass.

## 4. sft-v9 against sft-v7: the flips

There were 72 tasks with a different pass count:
- 2 → 0: 2 tasks;
- 0 → 2: 7 tasks;
- 1 vs 0 or 2: 63 tasks.

At 2 samples, nearly all of them are sampling variance in whether an episode bypassed the callables or made a path or
format slip. The readers found systematic differences only here:

**v9 better:**
- decision children's labels (sms-spam ham/spam, banking77 tally, abstract section stance);
- file destinations at the folder root and basenames in lines (bike-docks, fete-till);
- following `read_slip` (landing-ledger) and orphan handling (float-reconcile);
- calling the program's own `resolve` for stop evidence (schedule55);
- using the hinted `nl` lambda (decision_extract_chain).

**v9 worse:**
- destructive or mislocated file handling (lift-jobs, glassware, volunteer-caps);
- spec-faithful runs into the broken Neuralese chain (migration-tickets, costume-settle, referral-queue);
- more wrong_status and premature stops (12 vs 6, 12 vs 7);
- returning extra record fields;
- re-running child loops to the budget (snow-calls, hollowmere).

Net: +3 passes, which matches the measured neutral result (+0.6 pts, p 0.85).

## 5. Remedy map

| Class | Pool coverage (aei1 collection) | Remedy |
|---|---|---|
| Neuralese read-back (3.1) | Neuralese variants run in half of the groups; read-back is 13/465 | Bug hunt first (harness or representation); no data will fix a reader that receives nothing |
| Budget charged per decision sample (3.2) | n/a | Fixed (collector) |
| Decision label calibration (3.3) | ext:decisions 52 tasks, 121/212 | Training data: brief D |
| Spec misreads in reducers (3.4) | auth families 384/814; per-family 1–7 tasks each | Training data: brief A; harness items 3.8 |
| Child-call discipline (3.5) | composed 134/218; authored chains rare | Training data: brief B; teacher for deep chains |
| Codebase exploration (3.6) | natlang-codebase 35 tasks, 79/160 | Training data: brief C |
| Zero-rate families (3.7) | 0–2 episodes each | Teacher tier after 3.1; brief E |
| Runtime messages (3.8) | n/a | 5 fixed; the rest listed with owners |
| Prompt kernel | One suspect: `read_code` on a root says "For a judgment, decide it yourself … use eval only for exact work", which may nudge toward bypass (178, 194; uncertain) | Listed, not changed |

## 6. Not checked

- Whether any held-v3 failure is a Pop engine artefact. All three sets ran on Pop: no DGX re-run, no engine parity check.
- The cause of the Neuralese read-back loss (3.1). Only its size was measured, and only that the import is attached.
- The 22 unfinished held-v4 episodes, and held-v4 tasks not yet run.
- Successful episodes were not labelled, apart from those read for flips. The bypass rate among passes (64%) comes
  from child-call counts, not from reading.
- The readers' labels were not cross-checked by a second reader per episode. Four readers split their chunk further
  among sub-agents.
- Construct buckets come from keywords over free text.
- The tie-break rule of three-or-more-sample decision readouts.
- Whether the "cut off" footer, extra-field acceptance and `return_result` JSON-string refusal are by design.
