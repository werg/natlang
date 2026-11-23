---
name: natlang-authoring
description: Write, check, run, test and debug natlang programs — named `.nl` functions (typed YAML front matter plus natural-language instructions) with their callable folders, and inline `nl` calls in TypeScript. Use when a project has `.nl` files or a `natlang.json`, when the user asks for a natlang program, function or reducer, when `natlang check`/`call`/`test` reports an error, or when tuning, evaluating or optimizing a natlang program's instructions.
---

# Write natlang programs

Natlang is TypeScript with natural-language functions: typed functions whose bodies are instructions a model
carries out, checked like any other code. Use it to automate repeat agentic work on a small local model: the
program's types, data and exact steps are TypeScript, each judgment is a small typed function, and the runtime makes
finite judgments calibrated decisions, compiles repeated paths to code and checks every result against its type.

`natlang check` and the installed declarations are the authority. If this skill disagrees with them, trust the tool
and say so.

## The working loop

All commands are non-interactive. Run them from the program directory.

| Step | Command | What it does |
|---|---|---|
| Create | `natlang init DIR` | A runnable inbox planner: `types.ts` (`Untrusted`, `Is`), `route.nl` (`readout: decision`), `main.ts` (`decide` over the messages, an inline `nl<Reply>`, the `main` target), `examples.json`, `messages.json`, `natlang.json`, README, `AGENTS.md`. `--template classifier` writes the one-function `answer.nl` program. Never overwrites. |
| Check | `natlang check --json` | Types, `nl` signatures, callable-folder scope and loop policy, with no model. Fix every diagnostic before calling a model. |
| Call | `natlang call route.nl ARG... --json` | One real call. Arguments fill parameters in order: text as written, other types as JSON (`--inputs FILE` takes a JSON object keyed by parameter name). |
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
    assess/
      verdict.nl      its own helper: a typed decision (readout: decision), testable exactly
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
`args`, `returns`, `types`, `generic`, `kind`, `readout`, `model`, `uses`, `withhold`, `budget`; see
`args`, `returns`, `types`, `generic`, `kind`, `readout`, `model`, `uses`, `withhold`, `protect`, `budget`; see
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

1. **Write the types first.** Records and literal unions in `types.ts`, with `Untrusted<T>` on every value that comes
   from outside and `Is<T, "...">` on every property you would otherwise ask for in a sentence
   ([refinements](references/refinements.md)). Doc comments on fields reach the model: describe formats, units and
   what `null` means there.
2. **Exact steps are TypeScript.** Arithmetic, parsing, sorting, joining, counting, record assembly, I/O: functions in
   the callable folder or the host. A `.nl` function calls them. A store, API or verifier is a service the host
   passes to the runtime, not a helper the model can read and edit ([language](references/language.md)).
3. **Each judgment is one typed function.** Inline `nl<T>` in the code that needs it, or a `.nl` file when it is
   reused or tested on its own. When the instructions are several steps of judgment, number them and name the data
   each step uses. Children get distinct sub-questions and the inputs they need, and a parent gets its children's
   results, not their raw inputs.
4. **Finite answers are decisions.** `readout: decision` on a literal-union or boolean result: one scored pass, and
   `decide(fn, ...args)` returns the probabilities. Gate actions on confidence, and escalate the unsure cases with
   `fn.on({ model })`.
5. **Compose.** Map a judgment over a collection with `Promise.all`. Pass semantic functions as arguments. Rebind a
   function's context with `fn.in(context)`. Use `pluggable({ crisp, nl })` where a rule and a judgment compete
   ([composition](references/composition.md)).
6. **Iterate structurally.** `iterateOn(step, state).until(check)`, where the check can be code or `nl`; recursion
   only on a smaller argument; `for...of`, `for...in` over a record's keys, counted `for` loops whose bound is read
   once (`p <= store.pages()`) and array methods. There are no while loops. A TypeScript check needs
   `withMeasure(state => remaining)` or `withLimit({ maxSteps })`; a natural-language check needs neither. Add no
   turn, token or depth limits to make a run finish; a deployment that bounds its model declares
   `budget: { maxTurns, maxTokens }` on the functions that need more or less.
7. **Change files and code through transactions.** `folder.propose(reducer)`, check `proposal.folder` and
   `proposal.diff`, then `folder.accept(proposal)`; `folder.apply(reducer)` when success is the check. A verifier is
   a service, never a helper the model can edit.
8. **Close the loop.** `examples.json` holds the cases. `natlang specialize` compiles hot paths to crisp cases,
   `natlang tune` adapts the local model, and `natlang improve` searches source changes against the cases.

## How a call executes

The model receives the function's signature, types, instructions and inputs, then either answers directly (a text
or simple result) or works in a persistent TypeScript eval scope where parameters, captures, callable items and
services are bindings. Either way it answers with `return_result` (`success`, `blocked` or `failed`): a reply without a
tool call is not an answer. The result is checked
against the declared type; a mismatch goes back to the model to repair. A record or array result that `return_result`
receives as text (JSON or a JS object/array literal, read as data, never run) is accepted when the data it spells out
checks; otherwise the error names the field to fix or asks for the value written as an object or array. In host code a failed call rejects with
`NatlangCallError` (`callId`, `outcome`, `detail`, `trace`, and `reason`: `blocked`, `failed`, `type-mismatch`,
`repeated-failed-action`, `no-action`, `budget-exhausted`, `refinement`, `refused`, `incomplete`, `cancelled`, or
`error` when something other than the model's work ended it, which is then the error's `cause`).
`fn.on({ model: NAME })` is the same function on another of the runtime's models (what `model:` names), for one retry
on a stronger model without editing the source. `fn.in(context)` is the same function bound to another `Context`
(runtime package): a context that lacks an item the function's instructions name, or holds it with another signature,
is refused with `context-interface-mismatch`, which names the item; add it to the context or keep the function's own.
An inline `nl` rebound with `.in` keeps that context through `.with(...)`. Service calls and live-object writes are not rolled back; a
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
| `protected` (from `edit_code`) | The item judges the call (a `protect:` entry, `types.ts`, a `protect:` list): leave it as it is and fix the code that produces the value it rejects. |
| `capture-conflict` | Another task changed a captured `let` mid-call; if it recurs, make the state a parameter or a returned value. |
| `type-mismatch` / `undeclared-field` | The value or field doesn't match the declared type: fix the type or the instruction that produces it. |
| `iteration-measure-exhausted` | The measure hit 0 while `until` was false: make `until` true where the measure is 0. |
| `model_incomplete: repeated_failed_action` | The model repeated one failing action four times: clarify the instruction, split the stage, or use a stronger model. |
| `model_incomplete: no_action` | The model replied four times in a row without calling a tool or returning a result: say in the instructions exactly what to return, or use a stronger model. |
| Correct type, wrong answer | Improve the instructions, the evidence the call can reach, or the decomposition. Every run failing the same way points at the program, not the model. |

More diagnoses, and how to tell wiring evidence from model quality: [verification](references/verification.md).
Every error code: `spec/ERRORS.md` in the natlang repository.

## References

Read the one that matches the task:

- [composition.md](references/composition.md): the paradigm in working programs: mapped judgments, inline
  captures, `decide` floors and escalation, `.in(context)`, `pluggable`, `iterateOn` with a natural-language check,
  folder transactions, `Is`/`Untrusted`, Neuralese between stages, and the ladder from judgment to compiled code.
  Read before designing a program.
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
