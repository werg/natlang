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
| Check | `natlang check --json` | Types, `nl` signatures, callable-folder scope and loop policy, with no model. Fix every diagnostic before calling a model. Add `--idiom` (or `--idiom-strict`) to run the model-backed idiom reviewer (`idiom-review-*` warnings, or errors at confidence >= 0.8 under `--idiom-strict`). |
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
  types.ts            type aliases (`Untrusted<string>`, `Is<string, "...">`), visible below
  review.nl           the function: synthesizes the overall conclusion from the exact tally
  review/             its callable folder: what review.nl may call
    summarize.ts      exact TypeScript helper: runs assess concurrently with Promise.all and counts verdicts
    assess.nl         a natural-language helper
    assess/
      verdict.nl      its own helper: a typed decision (readout: decision), testable exactly
```

<!-- from: examples/review/review.nl -->
```yaml
---
description: Assess observations against a criterion, tally verdicts in code, and synthesize an overall conclusion.
args:
  observations: Untrusted<string>[]
  criterion: string
returns: Report
---
Call summarize(observations, criterion) to run the per-observation assessments concurrently and tally the verdicts.
Weigh the returned assessments and counts to write a one-sentence conclusion stating whether criterion holds overall
(naming any contradicting observation, or noting when no evidence supports it), and return the tally together with
conclusion.
```

`Untrusted<string>` marks outside text: the model sees it as quoted data, never as instructions. Types in front
matter are TypeScript type text read verbatim, so no YAML comments after them. A complete version is in
[assets/review/](assets/review/) (copy the whole directory). Front matter keys: `description`,
`args`, `returns`, `types`, `generic`, `kind`, `readout`, `model`, `uses`, `withhold`, `protect`, `budget`,
`recursive`, `measure`, `ensures`, `writes`; see
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

1. **Write the types first.** Records and literal unions in `types.ts`, with `Untrusted<T>` on every text people wrote
   (messages, notes, and names, titles and labels too) and `Is<T, "...">` on every property you would otherwise ask for in a sentence
   ([refinements](references/refinements.md)). Doc comments on fields reach the model: describe formats, units and
   what `null` means there.
2. **Exact steps are TypeScript.** Arithmetic, parsing, sorting, joining, counting, record assembly, I/O: functions in
   the callable folder or the host. A `.nl` function calls them. A store, API or verifier is a service the host
   passes to the runtime, not a helper the model can read and edit ([language](references/language.md)).
3. **Each judgment is one typed function.** Inline `nl<T>` in the code that needs it, or a `.nl` file when it is
   reused or tested on its own. When the instructions are several steps of judgment, number them and name the data
   each step uses. Children get distinct sub-questions and the inputs they need, and a parent gets its children's
   results, not their raw inputs. A judgment reads meaning, a whole passage when the meaning is spread over it, and
   maps words to referents (`{ side: 'right', ordinal: 2 }`); code counts, ranks and finds the item they pick.
4. **Finite answers are decisions.** `readout: decision` on a literal-union, boolean or finite-record result: the
   model reasons, then every value is scored, and `decide(fn, ...args)` returns the probabilities. Gate actions and
   escalate unsure or high-risk cases with `escalate(fn, { to: 'teacher', floor, risk?, policy? })` from
   `@natlang/node/std` (`policy: 'nl'` uses history-aware `shouldEscalate.nl`, reading prior escalation and disagreement
   rates from the call store). `readout: instant` scores without reasoning, in one pass, for calls where latency
   matters more than accuracy.
5. **Compose with `@natlang/node/std`.** Use the standard library's typed combinators instead of hand-rolling loops:
   collections (`semanticMap`, `semanticFilter`, `semanticPartition`, `semanticReduce`, `semanticMatch`,
   `semanticRank`), convergence (`semanticRepair`, `fixedPoint`, `semanticMerge`), policy (`escalate`, `withContext`,
   `compileRule`), and verified workspaces (`proposeAndVerify`, `branchAndSelect`). Pass semantic functions as
   arguments, bind leading args with `.partial(...)`, and use `pluggable({ crisp, nl })` where a rule and a judgment
   compete ([composition](references/composition.md)).
6. **Iterate structurally.** `iterateOn(step, state).until(check)`, where the check can be code or `nl`; recursion
   only on a smaller argument; `for...of`, `for...in` over a record's keys, counted `for` loops whose bound is read
   once (`p <= store.pages()`) and array methods. There are no while loops. A TypeScript check needs
   `withMeasure(state => remaining)` or `withLimit({ maxSteps })`; a natural-language check needs neither. Add no
   turn, token or depth limits to make a run finish; a deployment that bounds its model declares
   `budget: { maxTurns, maxTokens }` on the functions that need more or less.
7. **Change files and code through transactions.** `folder.propose(reducer)`, check `proposal.folder` and
   `proposal.diff`, then `folder.accept(proposal)`; `folder.apply(reducer)` when success is the check. A verifier is
   a service, never a helper the model can edit.
8. **Close the loop.** `examples.json` holds the cases. `natlang profile` reports call-store latency, tokens,
   confidence, escalation rate, and the recommended next optimization; `natlang specialize` compiles hot paths to crisp
   cases, `natlang tune` adapts the local model, and `natlang improve DIR --model MODEL` diagnoses recorded runs,
   rewrites instructions, gates on dev/confirmation splits, and installs with a rollback record.

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
| `forbidden-loop` (`` `while` loops are not allowed here ``, `not a checked finite counter loop`) | Rewrite as `for...of`, a counted loop with the counter alone against its bound (the message gives that form for your loop), an array method or `iterateOn`; it is reported at the loop. |
| `frontmatter-yaml` | Write the value it names in double quotes (a `description:` holding `: `). |
| `called itself without a smaller argument` | Recurse on part of the input, a shorter array or string, or a smaller non-negative integer. A `.nl` function declares `recursive: structural` to allow it. |
| `called itself, and its frontmatter does not declare recursion` | Declare `recursive: structural` (descent on part of an argument) or `measure: "<expression over the arguments>"` in the function's frontmatter. |
| `called itself with measure M, which is not smaller than N` | Make the self-call advance what the measure counts, or change the measure to the quantity that does fall. |
| `Revise the result so that ...` (an `ensures:` clause) | The result breaks a postcondition: produce one that meets the clause's message; past the repair budget the call fails `refinement-unsatisfied`, so say in the instructions how to meet it. |
| `unbound-name` | Use the call's admitted values or declare the value locally; the whole eval was rejected before effects. For a helper, check callable-folder nesting. |
| `invalid-binding` | A name reused for a result (`const plan = await plan(x)`): keep the result under another name (`planResult`). |
| `untrusted-instruction` | Pass untrusted text as an argument (``nl`Summarize the message.`(message)``), never interpolate it into instructions. |
| `protected` (from `edit_code`) | The item judges the call (a `protect:` entry, an item an `ensures:` check names, `types.ts`, the `protect:`, `ensures:`, `writes:`, `measure:` and `recursive:` lines): leave it as it is and fix the code that produces the value it rejects. |
| `writes-unsatisfied` | A directory reducer wrote an undeclared file or text that violates its `writes:` type: remove/restore the undeclared file or rewrite the file to its declared type (or update `writes:` outside the call). |
| `idiom-decision-readout` (warning) | Add `readout: decision` to the function's front matter so the model reasons and its finite result is scored without a tool loop; a function that must call helpers before it answers gets a companion folder for them. |
| `idiom-review-*` (`natlang check --idiom`) | Model-backed idiom findings (`mechanism-in-prose`, `guard-for-type`, `redundant`, `prose-interpreter`, `cue-word-rule`, `untyped-external-input`; errors at confidence >= 0.8 under `--idiom-strict`): move deterministic steps or keyword rules into TypeScript/pluggable crisp helpers, express constraints in types (`Is<T, P>`, `Untrusted<string>`), and keep each `.nl` function one narrow judgment. |
| `nl-not-tag` | Write the instructions in place as a template right after `nl` (``nl<T>`...`(x)``, or ``nl<T>(`...`, x)``); put changing values in `${...}` or pass them as arguments, never build the instruction text at run time, and never use `nl` itself as a value. |
| `readout must be decision (reason, then score the values), instant (score them without reasoning) or template` | Write `readout: decision`, or `readout: instant` for a hot-path call that should skip the reasoning. |
| `readout: decision needs a finite returns type` (or `readout: instant ...`) | Declare a finite `returns` (a literal union, `boolean`, or a record whose fields are all finite), or drop `readout:` and let the call run its tool loop. |
| `idiom-code-in-prose` (warning) | Move the step the instructions describe as code into a TypeScript helper in the function's companion folder, and let the instructions call it by name. |
| `idiom-fixed-cursor-iterate` (warning) | Map the step over the list (`await Promise.all(items.map(item => judge(item)))`) and keep `iterateOn` for steps that read the previous state. |
| `capture-conflict` | Another task changed a captured `let` mid-call; if it recurs, make the state a parameter or a returned value. |
| `function-signature-mismatch` | The function passed does not have the slot's function type: pass one whose parameters take what the slot passes, by position, and whose result fits (bind extra leading parameters with `fn.partial(...)`). |
| `type X is not in the natlang type grammar` (load) | Rewrite alias `X` in `types.ts` with records, unions, literals, arrays and `Record`, or keep it out of natlang signatures. |
| `type-mismatch` / `undeclared-field` | The value or field doesn't match the declared type: fix the type or the instruction that produces it. |
| `iteration-measure-exhausted` | The measure hit 0 while `until` was false: make `until` true where the measure is 0. |
| `model_incomplete: repeated_failed_action` | The model repeated one failing action four times: clarify the instruction, split the stage, or use a stronger model. |
| `model_incomplete: no_action` | The model replied four times in a row without calling a tool or returning a result: say in the instructions exactly what to return, or use a stronger model. |
| `iteration-unbounded` from `semanticRepair` or `fixedPoint` | Give `semanticRepair` its `attempts`; give `fixedPoint` a `maxSteps` with an exact `settled` test, or leave `settled` out so the shipped judgment decides. |
| `std-check-result` | Return the list of problems as strings from `check` or `verify`, and an empty list when the value passes. |
| Correct type, wrong answer | Improve the instructions, the evidence the call can reach, or the decomposition. Every run failing the same way points at the program, not the model. |

More diagnoses, and how to tell wiring evidence from model quality: [verification](references/verification.md).
Every error code: `spec/ERRORS.md` in the natlang repository.

## References

Read the one that matches the task:

- [composition.md](references/composition.md): the paradigm in working programs: mapped judgments, inline
  captures, `decide` floors and escalation, `.in(context)`, `pluggable`, `iterateOn` with a natural-language check,
  folder transactions, the standard library's repair, fixed-point, merge and verified or selected transactions
  (`@natlang/node/std`), `Is`/`Untrusted`, Neuralese between stages, and the ladder from judgment to compiled code.
  Read before designing a program.
- [language.md](references/language.md): front matter, callable folders, `uses:`, services, callable-folder
  TypeScript, types, eval rules, iteration. Read before creating or restructuring code.
- [patterns.md](references/patterns.md): orchestration, instructions for small models, iteration, reducers,
  checked stages, porting a system, delegation by access, decision floors, semantic pipelines over source records.
- [refinements.md](references/refinements.md): `Is<T, "predicate">` and `Untrusted<T>`.
- [verification.md](references/verification.md): what each kind of evidence establishes, scenario design, diagnosis.
- [fusion.md](references/fusion.md): writing stage chains the runtime can keep as Neuralese between stages.
- [neuralese.md](references/neuralese.md): `Neuralese<T>` soft state in `@natlang/node/std` (`carry: 'neuralese'`),
  `natlang profile`, and `natlang improve DIR`.
- [tuning.md](references/tuning.md): `natlang tune`: adapters so a program's calls run well on a small student.
- [adaptation.md](references/adaptation.md): evaluation suites and instruction optimization (`natlang eval`,
  `natlang optimize`).

To embed natlang in an application (runtime, model drivers, services, browser, persistence), use the
`natlang-integration` skill. Links into `spec/` and `docs/` resolve in a natlang repository checkout
(`natlang skills --json` prints where its skills live).
