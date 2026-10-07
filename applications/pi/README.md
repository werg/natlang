# pi on natlang

A port of [pi](https://github.com/earendil-works/pi)'s coding agent, as a natlang program. The agent is a
natural-language function on the big model. Its tools are natural-language functions on a small model, and so are
the quick judgments that help the big model do its job ("System One"). Crisp code only stands for the outside world
(a shell, the file system, the user) and runs the command line.

```
pi.nl                 the agent (model: big): pi's system prompt and rules; calls its tools in eval
pi/context.nl         AGENTS.md/CLAUDE.md up the tree, pi's skills (SKILL.md), files to start from
pi/context/scout.nl     the files a task most likely needs
pi/read.nl            pi's read: offset/limit, 2000 lines or 50 KB, continuation hint
pi/write.nl           pi's write: parents created, bytes written
pi/edit.nl            pi's edit: exact, unique, non-overlapping replacements against the original; then a review
pi/edit/review.nl       does the diff do what intent says? as-intended / unintended / incomplete
pi/bash.nl            pi's bash: a risk gate, then the run, tail truncation or a digest, full output saved
pi/bash/risk.nl         safe / review / destructive
pi/bash/digest.nl       long output cut to what the step needs
pi/progress.nl        progressing / repeating / stuck, every six actions
pi/done.nl            done / unfinished, once before answering
services.ts           shell.run, files.*, user.confirm: the outside world
```

The judgments with a finite answer use `readout: decision`, which scores every allowed value in one pass. The
functions that call them act on the probabilities with `decide(fn, ...args)` in eval. For example, bash refuses a
command at p(destructive) ≥ 0.7 and asks the user at p(review or worse) ≥ 0.5. These thresholds are part of the
instructions, like everything else.

pi's harness loop is the natlang interpreter's own loop. The big model reads tool results, and it can make several
calls per eval when the next steps are certain. It can also loop over many items with small-model `nl` judgments.
That covers what pi's codemode and routing did. The runtime moves old output into `transcript` when the context
fills up, and pi.nl asks for pi's checkpoint format in the note it keeps. The session log (`--session`, by default
under the state directory) holds one JSON line per finished call, with its full trace.

### The optimized harness

`agent.ts` and `tools.ts` keep a TypeScript version for comparison. It has the same judgments (imported as pi.nl's
children, `pi.bash.risk` and so on) around pi's loop and tools written in TypeScript. It adds codemode
(`harness/codemode.nl`: the big model hands natlang a script), routing (`harness/route.nl`: routine turns go to the
small model), and compaction into pi's checkpoint (`harness/compact.nl`). Use `--fast`.

## Running

```sh
natlang run applications/pi -- -p "Fix the failing test in src/paginate.js"      # print mode, the natlang program
natlang run applications/pi -- --big-endpoint http://host:8000 --big-model big "…"  # a separate big model
natlang run applications/pi -- --fast "…"                                          # the optimized harness
natlang run applications/pi -- eval [task...] --variants pure,system-one,plain
```

The launcher's model runs the tools and judgments. The big model runs pi.nl, and it is the same model unless
`--big-*` names another endpoint. The other options are:

- `--yes` approves commands that need review;
- `--max-turns N` limits the big model's turns;
- `--session FILE` sets the session log;
- `--skills DIR` adds a skill directory; `--skills skills` offers this repository's natlang authoring skills.

These apply with `--fast` only: `--no-system-one`, `--no-codemode` and `--route`.

`eval` runs each task in `tasks/` on a fresh git copy of its repository under each variant:

- `pure`: the natlang program;
- `system-one`: the optimized harness;
- `plain`: the TypeScript loop without judgments;
- `codemode` and `route`: the harness's extras.

A hidden check judges the result: the tests must pass, and test files must not be edited. The tasks are:

- js-off-by-one;
- py-parse-duration;
- noisy-suite: one failure in 445 lines of output;
- rename-api: many call sites;
- cli-flag;
- stale-build: deleting build output is allowed, but data must survive.

## Status

Both versions are tested with scripted models (`ts-host/test/pi.test.mjs`). The pure test checks the wiring: the big
model runs pi.nl and nothing else, the small model runs the tools, and their judgments act through `decide`. The live
evaluation is pending; results will be recorded here.
