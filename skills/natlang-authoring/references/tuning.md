# Tuning a program for a small student

`natlang tune DIR` fits per-definition adapters (and soft instruction states) so a program's calls run well on a
small trained student. Each call runs its own definition's adapter stack; source, call sites and types are unchanged.
Engine details are in [docs/TUNING.md](../../../docs/TUNING.md); models and profiles in
[docs/NATIVE_MODELS.md](../../../docs/NATIVE_MODELS.md).

## When tuning pays off

- The program runs often and has a stable job: routing, extraction, labelling, a fixed review step.
- The student gets it partly right untuned. A student that fails almost every root needs more examples, a teacher
  that succeeds on them, or a simpler decomposition first; tuning cannot learn from roots nobody solves.
- Correct results can be checked: an `expected` value, or a crisp verifier. Without checks, collection keeps only
  executions that complete, and validation reports completion rather than correctness.

Prefer fixing the program first when the failure is in the decomposition or the types: a stage that asks for too much
at once, an ambiguous type, an instruction that names the wrong thing. Tuning learns the job as written.

## Examples

`tune.examples.jsonl` beside the program holds root invocations, one object per line:

```json
{"function": "triage/route.nl", "input": {"ticket": "Refund for order 4411"}, "expected": "billing", "group": "customer-17"}
{"function": "triage/route.nl", "input": {"ticket": "Login loops after reset"}, "group": "customer-17"}
```

- `function` is the root definition's source path; `input` its arguments by name.
- `expected` is optional. With it the root has a direct oracle; without it the teacher's executions are kept when
  they complete, and evaluation counts completions.
- `group` names the source or world the example comes from. Examples of one group always land on the same side of
  the training, development and confirmation split, so related invocations never leak across it. Group by customer,
  document, repository or session, not by label.
- `"protected": true` holds an example out of every split for final reporting; nothing rewrites it.

Write examples from real inputs and cover the variety the program meets. The same input within one group counts once;
in different groups it is a distinct invocation.

## Examples from testing

Running the program is the cheapest source of examples:

- `natlang traces good CALL` adds a recorded root call with its output as `expected` (refused when another model
  than a tuning teacher or our own produced it);
- `natlang traces fix CALL --expected JSON` adds it with the answer you give;
- a call the student fails and the profile's `escalate` teacher reruns adds its input on its own, without
  `expected`.

Each session's examples form one group per tester and day. Then `natlang tune DIR` (first time) or
`natlang tune DIR --incremental` (after edits) refits. `natlang doctor --program DIR` lists what is waiting.

## Keep tuning across edits

A release binds adapters to definitions by identity. Editing a definition makes its adapter stale: the edited
definition runs untuned until `natlang tune DIR --incremental`, and the rest keeps its adapters. An inline `nl` site's
identity comes from its position unless you label it; label sites you want to keep their adapters through edits:

```ts
const lane = /* @natlangSite route-lane */ nl<Lane>`Route ${ticket} to a support lane.`;
```

Labels are unique per file. A renamed file goes in `tune.migrations.json` so its adapters carry over.

## Recovery

- `weights-unavailable`: an adapter is not loaded in the serving backend; rerun `natlang tune DIR` or restart the
  process.
- `weights-incompatible`: the release belongs to another base model; a tuned program runs on its own base
  automatically when that checkpoint is available (`natlang setup --program DIR` pins it), otherwise retune.
- `weights-stale`: a definition changed since the release; `natlang tune DIR --incremental`.
- "tuned at step X; PROFILE is at step Y": the student has trained further; `natlang tune DIR` retunes on it.
