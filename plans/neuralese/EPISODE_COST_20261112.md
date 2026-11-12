# Where base-policy episode tokens go, and how to cut them (2026-11-12)

CPU-only, read-only analysis of the Experiment 1 collection (`/home/werg/data/lfm3-rl-dgx-20261029/ops/exp1-rft/collect`,
base policy 5ff07638, 256 tasks x 8, temperature 1, 64 model requests, 3600 s wall budget per case). Snapshot taken
2026-11-12 ~17:30 CET while the collector was still writing. Nothing under the run directory was modified, apart from
the new `ops/exp1-rft/base-pass-rates.json`.

**Population.** 2048 queued entries: **1067 finished episodes** with model calls (421 task_success, 417
semantic_mismatch, 60 invalid_action, 16 budget_exhaustion, 153 wall-clock timeouts), 346 not started or still running, and
634 infrastructure failures. The 634 are actor or collector outages: 3 attempts each, under 50 s each, no result, 184k
tokens in total. They are excluded below. Outcomes come from `task_packet.classify_case` and `task_rollouts.reward_of`,
as in `task_rollouts.score`; finish status comes from the last `workers/*.a*.journal.jsonl` finish event.

**Token counts.** Per call I used `model_response.completion_tokens` and, for the think/action split, the captured
sampled ids (`rl_sampling`, expanded with `expand_captures`): thinking is everything before `</think>` (id 124902).
Timeouts and budget-exhaustion episodes have no `results.jsonl`, so their calls come from the latest
`jobs/*.partial.json` turns, with thinking counted by tokenizing `reasoning`. 732 host steps produce 0 tokens and are
not counted as calls. Totals:

- **18.66M generated tokens**, 17.5k per finished episode (success mean 9.8k, which matches the ~10.4k quoted).
- 20.1 calls and 870 tokens per call on average.
- 386 slot-hours of wall time, 414 including retried attempts.
- Prompt tokens are 12.4k per call and 257k per episode, about 14x the generated tokens. Prefix caching carries most of this.

The scripts are `extract.py`, `analyze.py` and `more.py` in the session scratchpad
(`/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/`, not committed). They re-run in
about 10 s.

## 1. Where the generated tokens go

### By outcome

| outcome | episodes | tokens | share | mean / median / p90 per episode | calls/episode | tokens/call | think share | mean wall s |
|---|---|---|---|---|---|---|---|---|
| semantic_mismatch | 417 | 7.01M | **37.6%** | 16.8k / 13.1k / 31.9k | 15.4 | 1091 | 89.7% | 1118 |
| timeout (3600 s wall, **unscored**) | 153 | 5.93M | **31.7%** | 38.7k / 36.6k / 65.5k | 45.2 | 856 | 90.0% | 3475 |
| task_success | 421 | 4.13M | 22.2% | 9.8k / 7.0k / 21.9k | 12.9 | 760 | 89.0% | 727 |
| budget_exhaustion (64 requests) | 16 | 1.01M | 5.4% | 63.2k / 54.9k / 90.7k | 113* | 558 | 90.5% | 2935 |
| invalid_action | 60 | 0.58M | 3.1% | 9.6k / 8.2k / 15.9k | 14.3 | 675 | 92.3% | 646 |

\* The partial snapshots of resumed attempts carry more turns than the 64-request budget.

**Only 22% of generated tokens are in successful episodes. 32% of tokens and 38% of slot-hours (148 of 386 h) are in
wall-clock timeouts, which `reward_of` leaves unscored, so they contribute nothing to RFT or GRPO.** These timeouts
are not caused by a slow server. Every one generated more than 16k tokens, 108 of 153 passed 32k tokens or 40 calls,
and they decoded at 11.1 tok/s per slot against 13.5 tok/s for successes. 53 of them are retried attempts after
earlier infrastructure failures. Sibling samples of the same task: in 92 of 153 cases all scored siblings failed, in
52 at least one sibling succeeded.

### Thinking vs action

89.5% of all generated tokens are thinking. Every call opens with `<think>`, and the action text is short (median
35 tokens, p90 170). Thinking per call has median 385, p90 1933, p99 5533 and max 8192 tokens.

- 76 calls hit the 8192-token per-call limit (`finish_reason: length`). They hold 2.7% of all tokens, and none of
  them produced an action (66 never closed `</think>`). That is pure waste.
- Calls over 4096 tokens: 561 calls holding 17.8% of tokens.
- Long thinking is not what separates failures from successes within a task. On the 75 tasks with mixed outcomes,
  successful samples think 995 tokens per call and failed samples 1008; failures think more on only 53% of tasks.
  Failures cost more because they make more calls (25.2 vs 18.1 per episode) and so generate more tokens (17.2k vs
  13.0k).
- Across tasks, success falls with thinking per call (quintiles 0.58, 0.51, 0.48, 0.39, 0.33), but this mixes in task
  difficulty: harder tasks get longer thinking.

### Repeated actions and loops

- **Loops are rare.** Three or more identical consecutive action sets in one invocation hold 0.06% of tokens (11
  episodes).
- **Repeated action sets** (the same set of tool names and arguments as an earlier call in the same invocation) hold
  5.4% of tokens and occur in 458 of 1067 episodes, including 134 of 421 successes. Experiment 1 drops those 134
  successes from its rows.

### Calls after the task was effectively solved

Successes stop promptly, so there is little waste here.

- 335 of 391 successes with a `return_result` were accepted on the first return. 43 of 391 needed 3 or more further
  calls after their first return attempt.
- In 129 file-editing successes, the last write is followed by a median of 1 verification call before the return
  (1.4% of those episodes' tokens).
- In 239 value-answer successes, the final value first appears in an action in the final call for 196 of them. 18.2%
  of these episodes' tokens come after the value's first appearance. This is an upper bound, because the value is
  often just read during exploration.

### Malformed actions and errors

**35.1% of tokens (in the 903 episodes with an action ledger) are in calls where at least one action was rejected,
errored or blocked.** 27% of successful turns and 30% of semantic_mismatch turns contain one. By tool:

| tool, outcome | share of tokens |
|---|---|
| eval rejected | 14.6% |
| read_code rejected | 5.0% |
| bash error | 4.5% |
| eval error | 3.8% |
| python error | 1.8% |
| read_file error | 1.2% |
| return_result blocked | 1.0% |

The largest classes of error message, by the tokens of the calls that hit them:

- `typescript-syntax` rejections of eval code, mostly Python written into the TypeScript eval: about 7% in all (4.5%
  for the generic message alone, the rest `';' expected`, `Unterminated template literal`, etc.).
- `bash cannot bypass the bash policy; use commands directly`: 294 errors, 1.9%. They come almost entirely from the
  two `natlang_codebase_repair` families.
- pyodide Python tracebacks: 1.8%.
- `invalid-binding: <input> is this call's input and already holds the caller's value`: about 1.5% summed.
- `read_code: no-such-function`: 0.4% or more.
- `transcript.search: bad-action, expected a scope-eval tool`: 68 errors.
- `search query must be nonempty`: 67.
- `read_page: No tool output was cut off`: 61.

**A real infrastructure bug.** 45 episodes on 11 tasks (`ext_codebase-{natlang,py}_codebase_*`: ts/py callers,
definition, effects, importers, raisers, subclasses) fail with `ENOENT: ... /home/werg/natlang/runs/natlang-rl-broad-external-20261031-v3/assets/repos/*.json`.
That directory does not exist on the DGX. Those families have 0/47 scored successes and 469k tokens (2.5%). These are
broken tasks, not policy failures.

Replies with no action at all (not counting 0-token host steps) hold 4.4% of tokens. Most of them are the
length-truncated calls above.

## 2. Tokens per call and calls per episode by task family

| family group | episodes | success/scored | timeouts | calls/ep | tokens/call | tokens/ep mean | tokens/ep median | share of tokens |
|---|---|---|---|---|---|---|---|---|
| auth_* (authored desks) | 320 | 58/254 | 66 | 19.9 | 1206 | 24.0k | 21.8k | 41.2% |
| ext_natlang-codebase repair (type_typo, ts_syntax) | 113 | 35/74 | 39 | **56.8** | 496 | 28.2k | 25.1k | 17.1% |
| curriculum_* | 97 | 40/81 | 16 | 22.5 | 849 | 19.1k | 13.2k | 9.9% |
| ext_bgkit_* reading | 61 | 29/51 | 10 | 16.6 | 1253 | 20.7k | 16.6k | 6.8% |
| ext_decisions_* | 101 | 43/98 | 3 | 21.1 | 545 | 11.5k | 8.1k | 6.2% |
| ext_composed_* | 54 | 27/41 | 13 | 15.5 | 1334 | 20.7k | 13.8k | 6.0% |
| ext_codebase-ops_* | 108 | 88/105 | 3 | 13.6 | 619 | 8.4k | 6.4k | 4.9% |
| ext QA/math (hotpot, 2wiki, gsm8k, babi, drop, countdown, ...) | 78 | 68/78 | 0 | 6.1 | 880 | 5.4k | 3.5k | 2.2% |
| ext_design-brief-gates_* | 49 | 20/48 | 1 | 3.7 | 1925 | 7.0k | 6.2k | 1.8% |
| ext_natlang-codebase (other) | 37 | 13/37 | 0 | 8.4 | 846 | 7.1k | 7.6k | 1.4% |
| ext_codebase-py/natlang codebase (**broken assets**) | 49 | 0/47 | 2 | 16.5 | 580 | 9.6k | 7.1k | 2.5% |

The single most expensive family is `natlang_codebase_repair_type_typo`: 57 episodes, 70 calls per episode, 35.7k
tokens per episode, 29 of the 153 timeouts, 10.9% of all tokens, 16/28 scored. It and `repair_ts_syntax` (43 calls per
episode, 6.2%) explore file by file at about 500 tokens per call. `auth_*` desks make fewer calls but think more
(1.2k tokens per call). The cheap, high-pass ext QA tasks finish in 1-6 calls. The full per-family table (128
families) comes from `analyze.py`.

Success-token p95 by group shows that a single global budget is wrong for the repair families:

| group | success tokens p95 | max |
|---|---|---|
| natlang-codebase repair | 54.6k | 55.4k |
| auth | 29.3k | 52.6k |
| bgkit | 28.2k | 62.6k |
| curriculum | 27.6k | 36.0k |
| composed | 24.2k | 36.2k |
| decisions | 20.1k | 26.7k |
| codebase-ops | 19.7k | 31.9k |
| natlang-codebase (other) | 16.2k | 21.6k |
| ext QA | 14.8k | 22.0k |
| design-brief | 8.5k | 12.2k |

## 3. When could failing episodes have been cut?

For successes, the total at completion gives the latest point at which a success happened:

- Successes complete within 7.0k tokens (p50), 11.7k (p75), 21.9k (p90), 27.6k (p95) and 51.8k (p99); the max is 62.6k.
- In calls: 9 (p50), 15, 27, 39 (p95), 87 (p99); the max is 128.
- In wall time: 553 s (p50), 1576 s (p90), 2153 s (p95), 3094 s (p99).
- Failures are longer: median 16.6k tokens and 16 calls, p90 43.0k tokens and 60 calls.

Each budget below is applied after the fact. Tokens saved are counted over all 18.66M; successes lost are those that
needed more than the budget.

| episode budget | successes lost | tokens saved |
|---|---|---|
| 16k tokens | 73 / 421 (17.3%) | 38.0% |
| 24k tokens | 35 (8.3%) | 23.1% |
| 32k tokens | 11 (2.6%) | 14.0% |
| 24 calls | 51 (12.1%) | 26.6% |
| 32 calls | 30 (7.1%) | 19.6% |
| 40 calls | 20 (4.8%) | 14.3% |
| 48 calls | 12 (2.9%) | 10.8% |
| **32k tokens or 40 calls** | **22 (5.2%)** | **19.2%** |
| 40k tokens or 48 calls | 13 (3.1%) | 13.4% |
| 24k tokens or 32 calls | 48 (11.4%) | 29.2% |
| 1800 s wall (for comparison) | 34 (8.1%) | 25.5% of wall |

**GRPO signal goes up under a budget, not down.** A cut episode becomes a scored 0, where today a timeout is unscored.
Counting tasks with outcome spread on current data:

| budget | tasks with spread | tasks with any success |
|---|---|---|
| none | 75 | 144 |
| 32k tokens | 85 | 140 |
| 24k tokens | 85 | 131 |
| 16k tokens | 82 | 120 |

At 32k tokens the extra spread comes from timeouts that become failures next to sibling successes. At 16k, lost
successes start to remove spread.

## 4. Per-task base pass rate

The collection runs sample-major and is incomplete, so most tasks have 3-6 of 8 samples scored (scored count: 1 → 18
tasks, 2 → 33, 3 → 50, 4 → 87, 5 → 57, 6 → 8). 253 tasks have at least one scored sample: 109 are all-0, 69 are
all-1 and 75 are mixed.

The 152 tasks with at least 4 scored samples split as:

| pass rate | tasks |
|---|---|
| 0 | 50 |
| (0, 0.1) | 0 |
| [0.1, 0.3) | 12 |
| [0.3, 0.5) | 6 |
| [0.5, 0.7) | 20 |
| [0.7, 0.9) | 21 |
| [0.9, 1) | 0 |
| 1 | 43 |

**59 of the 152 (39%) are in [0.1, 0.9].**

Tokens by task class:

| task class | tasks | share of tokens | tokens/episode |
|---|---|---|---|
| all-0 so far | 109 | **54%** | 22.4k |
| all-1 | 69 | 19% | 12.0k |
| mixed | 75 | 27% | 15.1k |

With 4 samples, a task whose true pass rate is 0.1 reads 0/4 with probability 0.66, so "all-0" overstates the
hopeless set. Even so, most compute goes to groups GRPO cannot learn from.

The per-task results are in `/home/werg/data/lfm3-rl-dgx-20261029/ops/exp1-rft/base-pass-rates.json`:

- Format: task_id → {successes, scored, mean_tokens, finished, timeouts, family}.
- scored = success plus verified failures. Timeouts and infrastructure errors are unscored.
- mean_tokens is the mean over finished episodes, including timeouts.
- Regenerate it when the collection finishes.

## 5. Recommendations

The savings below are measured on this data and do not multiply exactly, because they overlap.

1. **Replace the 3600 s wall budget with a generated-token episode budget, and score a budget stop as 0.** Use
   32k tokens or 40 policy calls for now, or per family at about 1.1x the family's success p95 (from section 2;
   repair families about 60k, design-brief about 10k).
   - Saving: 19.2% of generated tokens. The 153 timeouts (32% of tokens, 38% of slot-hours) become 108 earlier cuts
     that are scored.
   - Spread: tasks with spread rise from 75 to 85.
   - Risk: 22 of 421 base successes are lost (5.2%), mostly long repair and bgkit episodes. A per-family budget keeps
     most of them.
   - The harness already counts `completion_tokens` and requests per case (`max_model_requests`), so this is mechanism
     only and not model-facing. A wall budget should stay only as an infrastructure guard (for example 2x the token
     budget at the measured worst-case 11 tok/s per slot), and its stops stay unscored.

2. **Cap thinking.**
   - (a) Now: lower the per-call `max_tokens` from 8192 to 4096. The 76 calls that reach 8192 never act (2.7% of
     tokens). Calls between 4096 and 8192 that still act appear in 66 of 421 successes, so this cap is a policy
     change, and it should be measured on held-out before adoption.
   - (b) Propose, to be measured live: a thinking budget that forces `</think>` after 2048 thinking tokens and lets the
     model write its action.
     - Upper-bound saving: 14.9% (32.3% at 1024, 4.1% at 4096).
     - Within-task evidence that long thinking does not buy success: 995 vs 1008 thinking tokens per call, success
       vs failure.
     - Risk: 212 of 421 successes have at least one call that thinks more than 2048 tokens.
     - The forced token is not a policy sample, so the capture must mark it and its logprob excluded, as for
       host-authored actions.

3. **Stopping rules.**
   - Identical-action loops are not worth a rule: 0.06% of tokens.
   - "Stop at the 3rd repeated action set in an invocation" saves 14.8% and loses 37 successes (8.8%). At the 5th
     repeat it saves 10.1% and loses 20 (4.8%). Mostly redundant with rule 1; adopt it only if rule 1 is
     per-family and loose.
   - "Stop after 2 consecutive no-action replies" saves 2.3% and loses 22 successes. Not worth it once
     `max_tokens` is fixed.
   - Successes already stop promptly (section 1), so there is no post-solve rule to add.

4. **Task selection for GRPO: the largest lever.**
   - Only 27% of tokens currently land on tasks with spread. Selecting tasks with base pass rate in [0.1, 0.9] from
     `base-pass-rates.json` turns about 3.7x more of the generated tokens into learnable groups. Those groups are also
     cheaper: 15.1k tokens per episode against 22.4k for all-0 tasks.
   - Practical rule: keep tasks with at least 1 success and at least 1 failure, plus all-0 tasks with at most 3 scored
     samples (still uncertain). Hold 10-20% of each batch for unseen or all-0 tasks to keep exploring.
   - Re-estimate pass rates from every GRPO round's own groups; this costs nothing.
   - Drop the 11 broken `ext_codebase-{py,natlang}_codebase_*` tasks until their repo assets are restored.
   - Risk: only 59-75 eligible tasks today, which risks memorising them, and the ≤6-sample estimates are noisy.
     Finish the 8 samples for the borderline tasks first.

5. **Harness fixes.**
   - Not model-facing:
     - restore the missing `assets/repos/*.json` (or repoint the tasks): 2.5% of tokens and 11 tasks at 0%;
     - compute the generated-token budget in the collector;
     - mark forced thinking-close tokens in the capture.
   - Model-facing, to be measured live (about 48 samples per variant) before adoption, per AGENTS.md. Error-carrying
     calls hold 35% of tokens. The largest addressable classes:
     - Python written into the TypeScript `eval` (about 7%): make the error a one-sentence redirect to the `python`
       tool, or accept the common Python idioms.
     - The bash policy error (1.9%, repair families): say what the allowed form is instead of only refusing.
     - `invalid-binding` for redeclared inputs (about 1.5%): make `const x = x`-style shadowing a no-op, following
       "make standard JS idioms work".
     - `transcript.search` being offered and then rejected as not a scope-eval tool: don't offer it, or accept it.
     - Empty `search_files` queries (67), and `read_page` without a cut-off (61): make both harmless no-ops.
     - Repair families make 57 calls per episode at about 500 tokens per call while reading files one by one. A
       recursive listing or one grep tool result per folder is the obvious proposal to measure.

Expected combined effect: rules 1 and 2(a) together save about 20% of generated tokens per episode and turn
unscored timeouts into signal. Rule 4 roughly triples the share of compute that produces GRPO gradient. The thinking
budget (2b) and the error-message fixes (5) are the next 15-30% each, both pending live measurement.
