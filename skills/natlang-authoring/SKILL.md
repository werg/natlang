---
name: natlang-authoring
description: Write, check, run, test and debug natlang programs — named `.nl` functions (typed YAML front matter plus natural-language instructions) with their callable folders, and inline `nl` calls in TypeScript. Use when a project has `.nl` files or a `natlang.json`, when the user asks for a natlang program, function or reducer, when `natlang check`/`call`/`test` reports an error, or when tuning, evaluating or optimizing a natlang program's instructions.
---

# Write natlang programs

Natlang is TypeScript with natural-language functions. A `.nl` file is a function: YAML front matter declares its
typed signature, and the body is instructions a model carries out. Its typed result is checked like TypeScript. The
model that executes it may be small, so the author (you) puts the algorithm, the decomposition and the data contracts
in the source, and the interpreter only has to take the next step.

`natlang check` and the installed declarations are the authority. If this skill disagrees with them, trust the tool
and say so.

## The working loop

All commands are non-interactive. Run them from the program directory.

| Step | Command | What it does |
|---|---|---|
| Create | `natlang init DIR` | A runnable program: `answer.nl`, `natlang.json` (its `main` target), `examples.json`, README, `AGENTS.md`. Never overwrites. |
| Check | `natlang check --json` | Types, `nl` signatures, callable-folder scope and loop policy, with no model. Fix every diagnostic before calling a model. |
| Call | `natlang call answer.nl ARG... --json` | One real call. Arguments fill parameters in order: text as written, other types as JSON (`--inputs FILE` takes a JSON object keyed by parameter name). |
| Test | `natlang test --json` | Runs every case in `examples.json` (`[{ name?, function, inputs, expected }]`, compared exactly) and reports `{ ok, passed, failed, results: [{ expected, received \| error }] }`. Exits 1 if any case fails. |
| Debug | `natlang traces list`, `natlang traces show CALL_ID` | The recorded calls, and each step the model took in one: evals, tool results, child calls, the outcome. |
| Run | `natlang run [DIR] [-- ARGS]` | Builds the package and runs its target; `natlang dev` reruns on every edit. |

Model calls take seconds on a hosted model and can take minutes on a local CPU: check first, then call deliberately.
When you change behavior, add or update a case in `examples.json` and run `natlang test`.

**The model.** If a command reports that the local model runtime is not installed, ask the user before running
`natlang setup --yes` (or passing `--yes` to the command): a one-time download of llama.cpp and a model of about
1.8 GB. Alternatives: a hosted model with `--provider ID` on `call`/`test`/`run`
(`natlang models` lists providers), or any OpenAI-compatible server with `NATLANG_SERVER=URL NATLANG_MODEL=ID`.
A program can pin its model in `natlang.json` (`"model": { "profile": NAME }` or `{ "provider": ID, "model": ID }`).
`natlang doctor --json` reports the setup.

## A named function

```text
review/
  types.ts            type aliases, visible to the functions here and below
  review.nl           the function
  review/             its callable folder: what review.nl may call
    assess.nl         a natural-language helper
    summarize.ts      an exact TypeScript helper (default export)
```

```yaml
---
description: Assess each observation and count the verdicts.
args:
  observations: Untrusted<string>[]
  criterion: string
returns: Report
---
Assess every observation against the criterion with assess; the assessments are independent, so they can run
together. Then return summarize of the assessments, in the original order.
```

`Untrusted<string>` marks outside text: the model sees it as quoted data, never as instructions. Types in front
matter are TypeScript type text read verbatim, so no YAML comments after them. A complete version is in
[assets/review/](assets/review/) (copy the whole directory). Front matter keys: `description`,
`args`, `returns`, `types`, `generic`, `kind`, `readout`, `model`, `uses`, `withhold`; see
[language and source contracts](references/language.md).

**Callable folders.** A function calls exactly the items in its companion folder (`foo/` beside `foo.nl`), and their
children as properties. Items of one folder do not see each other. So when `inspect` must call `decide` during its own
call, nest it: `root/inspect.nl` and `root/inspect/decide.nl`. Two siblings `root/inspect.nl` and `root/decide.nl` give
`root` both, but `inspect` cannot reach `decide` (`no-such-function`). A helper that functions in different folders all
need is listed in each one's `uses:` instead of being copied.

**Inline `nl` in TypeScript.** For a focused judgment inside ordinary code:

```ts
import { nl } from '@natlang/node';
const priority: 'urgent' | 'normal' = await nl`Decide whether ticket is urgent.`(ticket);
```

The compiler plans every `nl` before the model runs: the annotation (or `nl<T>`) is the result type, the call's
arguments are its parameters, and variables the instructions mention by name are captured live. Build TypeScript
projects with `natlang build` (or `natlang run`) so `nl` is lowered. Inline calls in application code see the nearest
`natlang.d/` folder as their callable folder.

## Design the program

1. **Draw the boundary.** TypeScript does exact work: arithmetic, parsing, sorting, counting, I/O, bookkeeping.
   Natural language does judgment: interpreting, classifying, planning, choosing what to do next, explaining.
   Don't make the model do arithmetic in its head, and don't compute everything in code and ask the model only to
   approve. Most applications are ordinary code with a few `nl` decisions where meaning matters.
2. **Give each function one responsibility** with a precise signature. Split a stage that asks for several judgments
   at once into typed children, and give the parent the children's results, not their raw inputs. More files are not
   more depth: add a child when the parent needs its result.
3. **Put constraints in types, not sentences.** A literal union (`"billing" | "technical" | "other"`) or a record in
   `returns` beats "answer with one of...". A property a reader can judge from the value goes in
   `Is<string, "one line of at most 60 characters">`, and outside text (user input, pages, logs, service results)
   in `Untrusted<string>` instead of "never follow instructions in it". See [refinements](references/refinements.md).
   Doc comments on fields in `types.ts` reach the model: describe formats, units and what `null` means there.
4. **Write instructions a small model can follow.** Number the steps of a multi-step instruction and name the data
   each step uses and how to reach it (`facts.page(n)`). State what one loop step does and when it stops. Keep
   finishing words ("missing", "cannot", "not found") for real outcomes, or say when "not found" is the answer.
   Port an algorithm as its steps over named data, not as "do what tool X does". See [patterns](references/patterns.md).
5. **Make the outside world a service, not a helper.** A callable folder holds code the program owns, and the model
   can read and edit all of it (`read_code`, `edit_code`). A store, API, simulator, board or verifier is a service:
   the host passes it to the runtime with a declaration (what the model learns about it), callable-folder code imports
   it from `natlang:services`, and calls are traced effects. A stuck model will rewrite a "helper" that refuses its
   move. Never let correctness depend on the model not reading source.
6. **Loop structurally.** Callable-folder TypeScript and eval code have no `while`, `do`, open `for(;;)` or
   generators: use `for...of`, `for...in` over a record's keys, counted `for` loops, array methods, recursion on a
   smaller argument, or `iterateOn` for open-ended refinement
   (`iterateOn(step, initial).withMeasure(s => s.pending.length).until(s => s.pending.length === 0)`).
   Run independent calls together with `Promise.all(items.map(f))`.
7. **Use `readout: decision`** for a finite result (a literal union, `boolean`) that gates an action: one scoring pass
   instead of a tool loop, and callers can act on the probabilities with `decide(fn, ...args)`.
8. **Don't add arbitrary limits** (turns, tokens, depth) to make a run finish; long productive runs are intended, and
   budgets are runtime options for deployment.

## How a call executes

The model receives the function's signature, types, instructions and inputs, then either answers directly (a text
or simple result) or works in a persistent TypeScript eval scope where parameters, captures, callable items and
services are bindings, and finishes with `return_result` (`success`, `blocked` or `failed`). The result is checked
against the declared type; a mismatch goes back to the model to repair. In host code a failed call rejects with
`NatlangCallError` (`outcome`, `detail`, `trace`). Service calls and live-object writes are not rolled back; a
captured `let` is written back only after a successful eval. Every call is recorded in the machine's call store
(`natlang traces`); keep sensitive values out with `"recording": { "exclude": [...] }` in `natlang.json`.

## Common errors

| Error | Fix |
|---|---|
| `nl-unknown-return`, `nl-ambiguous-signature` | Annotate the binding or write `nl<T>`; don't widen to `any`. |
| `nl-undeclared-type` (in eval) | Declare the type in that eval's scope or write the structure in the annotation. |
| `no-such-function` from a child | The callee is a sibling, not in the caller's folder: nest it (`root/inspect/decide.nl`) or list it in `uses:`. |
| `` `while` loops are not allowed here `` (`callable-scope`) | Rewrite as `for...of`, a counted loop, an array method or `iterateOn`; the message gives the file and position. |
| `called itself without a smaller argument` | Recurse on part of the input, a shorter array or string, or a smaller non-negative integer; a function may not appear in its own chain of callers. |
| `invalid-binding` | A name reused for a result (`const plan = await plan(x)`): keep the result under another name (`planResult`). |
| `untrusted-instruction` | Pass untrusted text as an argument (``nl`Summarize the message.`(message)``), never interpolate it into instructions. |
| `capture-conflict` | Another task changed a captured `let` mid-call; if it recurs, make the state a parameter or a returned value. |
| `type-mismatch` / `undeclared-field` | The value or field doesn't match the declared type: fix the type or the instruction that produces it. |
| `iteration-measure-exhausted` | The measure hit 0 while `until` was false: make `until` true where the measure is 0. |
| `model_incomplete: repeated_failed_action` | The model repeated one failing action four times: clarify the instruction, split the stage, or use a stronger model. |
| Correct type, wrong answer | Improve the instructions, the evidence the call can reach, or the decomposition. Every run failing the same way points at the program, not the model. |

More diagnoses, and how to tell wiring evidence from model quality: [verification](references/verification.md).
Every error code: `spec/ERRORS.md` in the natlang repository.

## References

Read the one that matches the task:

- [language.md](references/language.md): front matter, callable folders, `uses:`, services, callable-folder
  TypeScript, types, eval rules, iteration. Read before creating or restructuring code.
- [patterns.md](references/patterns.md): orchestration, instructions for small models, iteration, reducers,
  checked stages, porting a system, delegation by access, decision floors, semantic pipelines over source records.
- [refinements.md](references/refinements.md): `Is<T, "predicate">` and `Untrusted<T>`.
- [verification.md](references/verification.md): what each kind of evidence establishes, scenario design, diagnosis.
- [fusion.md](references/fusion.md): writing stage chains the runtime can keep as Neuralese between stages.
- [tuning.md](references/tuning.md): `natlang tune`: adapters so a program's calls run well on a small student.
- [adaptation.md](references/adaptation.md): evaluation suites and instruction optimization (`natlang eval`,
  `natlang optimize`).

To embed natlang in an application (runtime, model drivers, services, browser, persistence), use the
`natlang-integration` skill. Links into `spec/` and `docs/` resolve in a natlang repository checkout
(`natlang skills --json` prints where its skills live).
