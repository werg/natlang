# Teacher execution setup

The teacher is Ternary Bonsai 2 27B PTQ1_0, served by llama-server on port
8081. The probes record the actual `/v1/models` response, program, prompt,
tool transcript, runtime outcome, effects, and audit results. Teacher probes
are quarantined under `runs/`; they never append training references.

## Keep the execution path simple

- Use `natlang/prompts/tools_teacher_compact.md`, temperature 0, low reasoning
  effort, and a 256-token thinking budget.
- Use `LlamaServerDecoder(json_text_values=True, tool_aliases={"call":
  "call_function"})` for this teacher/server combination. Text values are plain
  strings; other values are JSON text. Runtime validation still decides
  whether a proposal fits its destination. Invalid proposals remain possible.
- Keep compact state, caller validation feedback, and no self-review or local
  repair loop. Expanded state and review prompts remain student experiments.
- `done` explicitly signals successful completion. It checks return validity
  and closed numbered lines, and stops later calls in the proposed batch.
  `report_error` and `report_blocker` remain failure signals.
- Existing values can be copied with `write(source=...)`. The runtime preserves
  types, permits one incidental JSON quoting layer when necessary, and never
  unwraps a `{ "value": ... }` object. Fold/iterate literal initializers follow
  the same quote rule; source references retain their existing behavior.
- Missing optional inputs are explicitly identified both in the opening state
  and on direct reads. An actual empty string remains a valid supplied value.

These are interface clarifications and validation rules, not a deterministic
interpreter for the program's instructions. The model still chooses its actions.

## What the teacher tests exposed

| Run | Result | Finding |
| --- | --- | --- |
| `teacher-behavior-s881.json` | stopped after two cases | Teacher changed explicitly required Text into Num. |
| `teacher-behavior-prompt-s881.json` | six exploratory cases | A longer admonition did not fix wrong bindings or malformed values. |
| `teacher-behavior-compact-s881-rescored.json` | 14/16 | Missing write payload and wrong formal parameter name. |
| `teacher-behavior-final-s882-audit-v3.json` | 14/16, one wrong success | Missing input treated as empty; missing write payload. |
| `teacher-simple-focused-s883.json` | 5/6 | Explicit JSON-text transport and `done` worked; map duplicated the item input. |
| `teacher-simple-full-s884.json` | 18/21, one wrong success | Missing direct read still returned an ambiguous placeholder; map binding and quoted fold initializer failed. |
| `teacher-simple-focused-s885.json` | 4/4 | Missing versus empty, map, and fold pass after the targeted fixes. |
| `teacher-simple-full-s886.json` | 19/21, one wrong success | Teacher redirected an impossible direct return into a record field; another case attempted a premature error with truncated tool arguments. |
| `teacher-destination-effects-s886.json` | 8/8, zero wrong successes | Revised prompt passes both affected failure cases and their valid controls, on the original group and one fresh group. |

Schema isolation probes reproduced unexpected `{ "value": ... }` objects with
an untyped tool parameter. Explicit JSON shapes improved scalar handling but
did not reliably handle heterogeneous payloads. An explicit string transport
for JSON values fixed the tested record/list cases without a compatibility
unwrapper. This is evidence about the teacher/server interface; it does not
identify the model, parser, or constrained generation as the sole cause.

The audit was also corrected: a faithful attempted operation that encounters a
genuine type conflict is an acceptable failure. The model need not anticipate
every conflict by calling `report_error`. Redirecting the call, supplying a
different value, or reporting success still fails. Equivalent numeric fold
initializers are accepted whether passed by reference or literal.

After s886, the prompt and tool description explicitly state that an exact
destination cannot gain a field suffix to make types fit. The prompt also
clarifies ordered execution before a later conflict and requests a short error
explanation. The earlier field-copy example was removed from the general write
description. These changes address observed failures; their effectiveness must
be measured rather than assumed. The full s886 run used the preceding prompt,
preserved verbatim in its artifact.

The targeted follow-up passed all eight cases. Impossible direct bindings
attempted the requested destination and failed validation; valid bindings
succeeded. Required effects occurred before the later reported type conflict,
and the matching success cases completed. This addresses the observed failures
on those instances; the latest wording has not had another complete 21-case
run or a broad unseen-template evaluation. Keep per-sample admission checks.

## Reproduce and inspect

```bash
.venv/bin/python scripts/teacher_behavior_probe.py --seed 886 --temperature 0 --reasoning-effort low --json-text-values --system-file natlang/prompts/tools_teacher_compact.md --out runs/teacher-new-check.json
.venv/bin/python scripts/teacher_leaves.py --families cb_shopkeeper cb_webserver --n 4 --seed 886 --limit 4 --dry-run --audit-out runs/teacher-leaves-new-check.jsonl
```

Use fresh output paths. `teacher_leaves.py` uses this setup by default, records
every accepted/rejected attempt, and only admits completed outputs that pass
the existing leaf checks. `--dry-run` performs the same checks without admission.
Those checks combine crisp constraints and model judgments; they are not proof
of semantic correctness. A passing fixture suite likewise does not establish
that all teacher-generated programs are trustworthy.

The student behavior corpus is documented in [AGENT_SUPPORT.md](AGENT_SUPPORT.md).

The real collector dry run `runs/teacher-leaves-simple-s886.jsonl` passed all
four selected shopkeeper leaves with the revised prompt, including its crisp
checks and teacher judgments; zero references were admitted. This verifies
that collection path for those examples, not all generative leaf families.
