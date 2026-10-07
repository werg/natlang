# pi on natlang

A port of [pi](https://github.com/earendil-works/pi)'s coding agent. The big model works as it does in pi: pi's
system prompt (with AGENTS.md/CLAUDE.md context files) and pi's four tools. `read` truncates to 2000 lines or
50 KB, `write` creates parent directories, `edit` makes exact, unique, non-overlapping replacements, and `bash`
tail-truncates output and saves the full text. Sessions are JSONL, and compaction uses pi's checkpoint format.

What natlang adds is **System One**: small-model natural-language functions that help the big model. The finite
judgments use `readout: decision`, which scores every allowed value in one pass. The host acts on their
probabilities through `runtime.decide`, with a floor per action:

| When | Function | Effect |
|---|---|---|
| before each `bash` (and each `shell.run` in codemode) | `system1/risk` safe / review / destructive | refuse at p(destructive) ≥ 0.7; ask the user at p(review or worse) ≥ 0.5 |
| long `bash` output (> 80 lines) | `system1/digest` | errors, failing tests and summary lines for the current step; full output saved to a file |
| after `edit`/`write` | `system1/review` as-intended / unintended / incomplete | a bracketed note when the diff doesn't match what the model said |
| every 6 tool calls | `system1/progress` progressing / repeating / stuck | a steering note |
| final answer | `system1/done` done / unfinished | sent back once when judged unfinished |
| start | `system1/scout` | the files to start from |
| context full | `system1/compact` | pi's checkpoint summary |
| before each turn (opt-in) | `system1/route` routine / hard | p(routine) ≥ 0.8 gives the turn to the small model |

**codemode.** The big model can hand natlang a TypeScript script and read only what it reports. The script can
use `shell.run` (risk-gated), `files.*`, and typed `nl` judgments that run on the small model. The script runs in
natlang's eval. A wrapped driver answers the `codemode.nl` call by evaluating the script, and every other request
goes to the small model.

## Running

```sh
natlang run applications/pi -- -p "Fix the failing test in src/paginate.js"     # print mode
natlang run applications/pi -- --big-endpoint http://host:8000 --big-model big "…"  # a separate big model
natlang run applications/pi -- eval [task...] --variants plain,system-one,codemode,route
```

The launcher's model runs System One. The big model is the same model unless `--big-*` names another endpoint.
Other options are `--no-system-one`, `--no-codemode`, `--route`, `--yes` (approve commands that need review),
`--max-turns N` and `--session FILE`.

`eval` runs each task in `tasks/` on a fresh git copy of its repository under each variant. A hidden check judges
the result: the tests must pass, and test files must not be edited. The tasks are js-off-by-one, py-parse-duration,
noisy-suite (one failure in 445 lines of output), rename-api (many call sites), cli-flag and stale-build (deleting
build output is allowed; data must survive).

## Status

The tools, `runtime.decide`, every System One hook, codemode and routing are tested with scripted models
(`ts-host/test/pi.test.mjs`). The live evaluation is pending; results will be recorded here.
