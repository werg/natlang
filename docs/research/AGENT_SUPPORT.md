# Agent support and review experiments

> **Status (2026-09-23):** historical. These experiments ran on the Python runtime and its probes, which have been removed; the findings stand as recorded evidence. Current prompts and tools are in `ts-host/src/native/`.

2026-09-19, v8 350M interpreter. No checkpoint was trained during these experiments.

## Implemented behavior

- `ToolSurface(state_view=True)` adds actual inputs, typed locals, return contents
  and missing fields, current line marks, and available function signatures to
  the opening observation and successful tool results. Data stays in tool
  messages. It does not parse the program to choose actions. **Off by default**:
  the combined expanded view regressed on the current checkpoint.
- Existing `call(..., done=N)` and `write(..., done=N)` already validate marks
  before execution and close lines only after success. We use these in the new
  references rather than inventing another tool. The schema now caps ranges at
  two endpoints, matching runtime validation. Standalone marks remain available.
- Existing `write(path=..., type=..., source="let/result", done=N)` copies a
  computed value without regenerating it. Reference trajectories demonstrate
  calling first, then copying in a later turn. Expanded prompts emphasize
  keeping the requested destination and not marking unexecuted work complete.
- Original runtime prompts remain the default. The opt-in prompts are
  `natlang/prompts/tools_small_support.md` and `tools_delegate_support.md`.
- `review_scope="actions"` reviews calls, standalone completion marks, edits,
  source copies, and actions carrying `done`, even without value logprobs.
  Low-confidence value review can run alongside this. Review stays optional.
- Guided review has four choices: `approve`, `withdraw`, `error`, `blocker`.
  Withdrawal means the proposal is wrong but the task remains executable.
  Error means the task is impossible as written; blocker means information is
  missing. The legacy tool name `review_write` now covers structural actions too.
- `withdrawal_policy="caller"` is the default. `"retry"` permits **one** fresh
  proposal per episode from unchanged state. It adds a neutral withdrawal notice
  to the main history, without the rejected action or reviewer explanation.
  A second withdrawal returns to the caller. Reviews and retries share the same
  episode budgets. Earlier completed effects remain; no pending-batch effects
  occur before all its reviews approve.
- `review_prompt` can be `baseline`, `repeat_instructions`, or `checklist`.
  The reminder repeats the original user program verbatim in the review fork.
  The checklist additionally asks for brief evidence about the selected branch,
  source/destination, result, and completion marks. Normal execution history is
  unchanged by an approved review. Baseline remains the default.
- Proposed batches retain their pre-action messages for inspection/replay.
  `scripts/review_probe.py` evaluates saved labeled proposals under multiple
  orders/prompts, without executing any action. Input and prompt-source hashes
  identify the comparison. No runtime orchestration policy is inferred from text.

## Live execution comparison

Sixty odd-group instances from the existing matched templates (seed 992),
caller validation feedback, temperature zero. These are familiar templates, not
novel-program validation. All four runs below used the expanded delegate prompt;
the state view is the difference between the first two rows.

| Configuration | Correct / 60 | Reviews | Model turns | Completion tokens |
| --- | ---: | ---: | ---: | ---: |
| Expanded prompt, original compact state | 47 | 0 | 137 | 1,897 |
| Expanded prompt and full state | 28 | 0 | 233 | 6,174 |
| Full state, action review, withdrawal to caller | 31 | 80 | 222 | 4,366 |
| Full state, action review, one reconsideration | 32 | 96 | 254 | 5,049 |

Action-review rows also used value threshold 0.994690001059068. Expanded state
repeats workspace, marks, and function signatures after actions and supplies an
opening observation even for input-free tasks. This combined intervention is
harmful here; this run does not isolate token volume, individual components, or
training-distribution mismatch. Failures included repeated writes after a valid
return, fabricated missing evidence, skipped calls, and exhausted budgets.
Do not interpret these runs as proof that extra state or self-review helps.
The original compact state and original runtime prompts remain defaults.

Artifacts: `runs/agent-support-{no-state,state,actions-caller,actions-retry}.json`.
Those runs predate the final CLI defaults: their `no_state_view` argument is the
inverse of the current `state_view` flag. Their exact system prompts are stored.
The earlier 52/60 baseline used the original prompt and is a separate run.

## Identical-proposal prompt comparison

94 saved proposals from two separate instance groups, seed 99105, expanded
reference histories. There are 34 good proposals and 60 bad ones: 48 should be
withdrawn, ten task errors, and two missing-information blockers. Bad numeric or
text writes are type-preserving near misses. Each prompt sees the same saved
pre-action history and proposal; all use reason before verdict.

| Review prompt | Exact verdict / 94 | Bad proposals approved / 60 | Good proposals rejected / 34 |
| --- | ---: | ---: | ---: |
| Baseline | 36 | 56 | 2 |
| Repeat instructions | 41 | 42 | 6 |
| Repeat plus checklist | 43 | 29 | 19 |

The instruction reminder increases detection but also false rejections. The
checklist is too aggressive for this checkpoint. Correctly refusing a proposal
but calling it withdrawal rather than task error still fails the exact-verdict
metric. No prompt variant is promoted to default. Results apply to these saved
reference contexts, not a measured improvement in end-to-end execution.

Artifacts: `runs/agent-support-review-eval-s99105.jsonl` and
`runs/agent-support-review-prompts-s99105.json`.
An earlier order-only comparison on seed 99104 used more obvious wrong values:
reason-first 38/94 versus decision-first 36/94, both approving 53/60 bad proposals.
It is a separate dataset and should not be pooled with the reminder comparison.

## New training references

Review experiment batch: `data/agent-support-s108.jsonl` and
`data/agent-support-reviews-s108.jsonl`, with
`data/agent-support-s108.manifest.json`.

- 1,600 executed programs across 100 groups; 3,300 verified execution turns.
- 4,700 oracle-labeled review turns: 1,700 approve, 2,400 withdraw, 500 error,
  100 blocker. These are reference labels, not student-generated judgments.
- Matched count versus element selection; direct versus wrapped return bindings;
  evidence present versus missing; valid versus impossible bounds and branches;
  calls with prior effects; numeric and text collections; flat and nested records;
  shipment reporting and support triage contexts with varied phrasing.
- Correct trajectories use successful-action marks and reference copies.
  Wrong proposals include early completion marks, changed destinations, and
  type-preserving wrong values. Correct proposals are explicit review controls.
- All related variants and execution/review turns share `program_id` per group.
  Split by that identifier. Review prompts rotate across the three variants.
- Compact state is used in this batch. `--state-view` generates experimental
  expanded-state histories; both modes use the support prompt. Runtime executes
  every execution reference and verifies result/failure and effects. Review
  labels follow fixture construction; their candidates are not executed.
- Intermediate s104/s105/s106 artifacts are development batches; use s108 for
  the documented dataset. No new model training has been launched.

```bash
.venv/bin/python scripts/generate_agent_support.py --groups 100 --seed 108 --out data/agent-support-s108.jsonl --reviews data/agent-support-reviews-s108.jsonl
# Use fresh output paths: generation refuses overwrite.
.venv/bin/python scripts/review_probe.py --input runs/agent-support-review-eval-s99105.jsonl --out runs/new-reminder-comparison.json --orders reason_first --prompts baseline repeat_instructions checklist
.venv/bin/python scripts/confidence_probe.py --groups 10 --seed 992 --partition test --support-prompt --state-view --review-scope actions --withdrawal-policy retry --review-prompt repeat_instructions --careful-threshold 0.994690001059068 --out runs/new-action-review.json
```

Tests cover successful-action marking, rejected calls leaving marks open,
source copying, factual state, withdrawal without mutation, one-reconsideration
limits, structural review without logprobs, instruction-reminder isolation,
reference execution, grouped review pairs, and review confusion counts.

## Execution-only student expansion

The newer `data/agent-behavior-s111.jsonl` contains 33,000 verified execution
turns across 1,000 groups (16,000 programs). It omits the review examples and
uses compact state. Half the groups finish with the explicit `done` tool; half
use the existing final-reply convention. Keep groups together when splitting.
The accompanying manifest records generation settings and source hashes.

SHA-256: `3495591d25e701d09cdd78204f9ec4bc7b51921ffe9789a293d31afb4d33b85a`.
The corpus is local and reproducible; it has not been used for a new training run.
Teacher findings and the simple collection configuration are documented in
[TEACHER_SETUP.md](TEACHER_SETUP.md).

## Honesty, persistence, and calibration labels

[LEARNING_LESSONS.md](LEARNING_LESSONS.md) tracks teacher and student difficulties
and their corresponding contrasts. The review generator now pairs justified
actions with wrong actions under the same skeptical check-up wording. It also
includes correct error/blocker reports, unsupported failure claims on feasible
tasks, premature error reports before required effects, and the teacher's
observed destination redirection. Reviews carry proposal-validity labels and
lesson IDs, without invented confidence probabilities.

Use `data/agent-honesty-s113.jsonl` and `data/agent-honesty-reviews-s113.jsonl`
with their manifest for this expansion. The s112 development batch is quarantined
under `runs/quarantine/`, outside the data directory, and is not for
training: an error-report contrast had an inaccurate feasibility explanation
for the prior-effect case, corrected in s113 and covered by a regression test.

The s113 batch has 33,000 executed reference turns and 71,000 review turns:
23,000 approve, 41,000 withdraw, 6,000 error, and 1,000 blocker. It contains
1,000 split groups. Review labels are constructed oracle contrasts, not
teacher-generated judgments or measured probabilities. No training has run on
this batch yet. Its manifest records both file hashes and source hashes.

```bash
.venv/bin/python scripts/generate_agent_support.py --groups 1000 --seed 113 --workers 4 --out data/agent-honesty-s113.jsonl --reviews data/agent-honesty-reviews-s113.jsonl
```
