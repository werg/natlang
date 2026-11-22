# WS4: The front door teaches the ambitious paradigm now, by example and by errors

**Parent:** [Campaign index](README.md). **v2** (2026-11-23). v1 scheduled the front door for "Wave 1-3" and its
corpus "quarantine". In v2 the front door ships in Wave A with exact content (below). The legacy corpora are migrated
or deleted (WS7).

Consumers are people and coding agents writing natlang programs. What they read: `natlang init` and the `AGENTS.md` it
writes, `skills/natlang-authoring` and `skills/natlang-integration`, `README.md`, `website/`, the playground, and the
CLI's errors. Each surface must carry the same stance, shown in working code:

> Hand natlang the repeat agentic work you would give a frontier agent loop. Types, data and exact steps are
> TypeScript. Every judgment is a small typed natural-language function, which a local 1.7B model runs, scored and
> calibrated when the answer is finite. The program checks its own results. It escalates only the cases it is unsure
> of, compiles its repeated paths to plain code, and improves itself from its own runs.

**Rules for every surface.** (1) Show the paradigm in a runnable example before explaining it. (2) Never show a wrong
form (AGENTS.md "Teach in errors, not prompts"; owner point 9). A skill or AGENTS.md is a prompt to the coding agent
that reads it. v1's ❌/✅ anti-pattern gallery is dropped. Wrong forms are taught by `natlang check` and the idiom
reviewer at the moment they are written. (3) Every command and API named exists at the time the text ships. A
construct still being built (std combinators, `--idiom`, `profile`) enters the text in the same push that ships it.
(4) The front-door text is also the teacher prompt for WS5's idiomatic-program demonstrations, so users and the
training data learn the same thing.

## 1. Inventory

| Surface | File | Today (verified 2026-11-23) | v2 item |
|---|---|---|---|
| `natlang init` | `ts-host/src/package/authoring.ts:26-125`; CLI `cli/commands.ts:984-994` (takes no flags) | one `answer.nl` 3-label billing classifier with `readout: decision` (chosen because it ran in 18.7 s instead of 136 s on local LFM3); generated AGENTS.md says "Exact work ... belongs in TypeScript; judgment belongs in instructions" and "Number the steps of a multi-step instruction" | F1 |
| Generated `AGENTS.md`/`CLAUDE.md` | same file, `agentGuide()` L79-125 | loop, exit codes, model setup, six writing bullets | F1 (§3) |
| `natlang new` stub | `createFunction` L131-140 (dashboard only) | "Number the steps if there are several" | F1 |
| Authoring skill | `skills/natlang-authoring/SKILL.md` (14 KB) + `references/{language,patterns,refinements,verification,tuning,adaptation,fusion}.md` + `assets/review/` | L8-11 "the author puts the algorithm ... in the source"; L90-93 "Most applications are ordinary code with a few `nl` decisions where meaning matters"; no `.in`, `pluggable`, Folder transactions, `decide`, `escalate`, `specialize`, `improve` in SKILL.md | F2 (§4) |
| Integration skill | `skills/natlang-integration/SKILL.md` + 9 references | "Choose the smallest adequate boundary" | F3 |
| README | `README.md` | strong first snippet (triage), then a classifier walkthrough | F4 (§5) |
| Website | `website/index.html` (9 chapters), `app.js` | hero `classify.nl`/`triage.ts` + `is_urgent.nl` exhibit; app cards link the treaty-obligation ledger (`index.html:707`) | F5 (§6) |
| Playground | `ts-host/playground/examples/{natural,interfaces,crisp,builders}.mjs`, `examples.mjs` | gallery = 5 featured natural examples + interface examples; *(v2 fact)* the 21 TS-only examples are test fixtures, not in the gallery (`examples.mjs:4`) | F6 (§7) |
| Errors | `spec/ERRORS.md` (catalog), inline messages (`compiler/inline.ts`, `runtime/kernel.ts`, `runtime/context.ts`, `native/return-errors.ts`), skill "Common errors" table | good one-sentence errors exist; idiom floor does not | F7 (§8) |

## 2. F1: `natlang init` writes a semantic-FP program (2 days)

`natlang init DIR` writes the program below by default. `natlang init DIR --template classifier` writes today's
single-file program (the dashboard Playground keeps calling `answerSource(custom)`). The template is the reasoned
default. There is no undecided menu (owner 2026-11-18 on modes).

**Speed constraint.** The first `natlang run` on local CPU must finish in under ~60 s after setup. The sample input
therefore takes the decision path plus at most one generative call. The executor measures it; if it is slower, the
sample is changed (a message routed `fyi`, which makes no generative call). The paradigm stays.

Files (`inbox/` is the program name; `natlang init` substitutes the directory name):

`types.ts`
```ts
/** One incoming message. The body is outside text: the runtime fences it and never follows instructions in it. */
export type Message = { from: string; subject: string; body: Untrusted<string> };
/** What the message needs from its reader. */
export type Route = 'reply' | 'fyi' | 'ignore';
/** A drafted reply: short, polite, and promising nothing the message does not ask for. */
export type Reply = Is<string, 'a polite reply of at most three sentences that makes no commitment the message does not ask for'>;
export type Plan = { from: string; subject: string; route: Route; confidence: number; reply?: Reply };
```

`route.nl`
```yaml
---
description: Decide what a message needs from its reader.
args:
  message: Message
returns: Route
readout: decision
---
reply when the sender asks a question or asks for an action; fyi when it informs without asking anything; ignore
when it is automated, promotional or a notification nobody needs to read.
```

`main.ts`
```ts
import { nl } from '@natlang/node';
import route from './route.nl';
import type { Message, Plan, Reply } from './types.js';

/** Plan the reader's inbox: route every message, draft replies for the ones that need one. */
export async function plan(messages: Message[]): Promise<Plan[]> {
  // Independent judgments run together; each is one scored decision on the local model.
  const decisions = await Promise.all(messages.map(message => route.decide(message)));
  const draft = nl<Reply>`Draft a reply to message for its reader.`;
  return Promise.all(messages.map(async (message, i) => {
    const { value, confidence } = decisions[i]!;
    const plan: Plan = { from: message.from, subject: message.subject, route: value, confidence };
    if (value === 'reply') plan.reply = await draft(message);
    return plan;
  }));
}
```
The executor checks whether `route.decide(message)` or a runtime `decide(route, message)` is the host-code form that
exists today. It writes whichever does. If neither exists in host code, adding one host-callable form of `decide` is
part of F1 (one implementation: the eval builtin's `runtime.decide`).

`main/` holds nothing at first. `examples.json` has three `route.nl` cases (one per label) and one `plan` case whose
expected value checks route and the presence of `reply` (exact). `natlang.json` targets `main.ts`'s `plan` with a
bundled `messages.json` (3 messages), `model.stock` the local default, and a `warmup` on `route.nl`. Generated
README: the command block as today, then one paragraph per file.

## 3. F1: the generated `AGENTS.md` (exact text)

```markdown
# Working on this natlang program

This program automates a piece of judgment work with a small local model. TypeScript holds the types, the data and
every exact step; each judgment is a typed natural-language function the model runs. Read `main.ts` first: it is the
whole program in one screen.

## Files

- `types.ts`: the data. `Untrusted<string>` marks outside text (the model reads it as data and never follows it);
  `Is<string, "...">` is a property the runtime checks on every value and repairs when it fails.
- `route.nl`: a named natural-language function. YAML front matter declares its typed signature; the body says what
  to decide. `readout: decision` makes the model score each allowed value in one pass: fast, and calibrated
  (`decide` gives you the probabilities).
- `main.ts`: TypeScript that maps `route` over the messages, and an inline function, ``nl<Reply>`Draft a reply to
  message for its reader.` ``, called only where a reply is needed. Inline `nl` sees the variables its instructions
  name.
- `examples.json`: cases with expected results; `natlang test` runs them on the model.
- `main/`: the callable folder: helpers `.nl` functions here may call (`.ts` for exact steps, `.nl` for judgments).
- Generated (do not edit): `*.d.nl.ts` and `.natlang/`.

## The loop

1. `natlang check --json`: types, signatures and call sites, without a model. Fix every diagnostic first; each one
   says what to write instead.
2. `natlang call route.nl '{"from":"a@b","subject":"Invoice","body":"Can you resend it?"}' --json`: one real call.
3. `natlang test --json`: every case in examples.json. When you change behaviour, add a case first.
4. `natlang traces show CALL_ID`: every step the model took. `natlang traces fix CALL_ID --expected JSON` turns a
   wrong result into a saved case.

Exit codes: 0 ok; 1 the program is wrong; 2 usage error; 3 the model is unavailable.

## Growing the program

- A new judgment is a new typed function: a `.nl` file when it is reused or has its own cases, an inline `nl<T>`
  when it belongs to one place in the code. Give it the narrowest result type that fits: a literal union with
  `readout: decision` when the answer is one of a few values, a record when it has parts.
- An exact step (counting, sorting, parsing, joining records, arithmetic) is a TypeScript function; a `.nl` function
  that needs it calls it from its callable folder.
- A property of a result goes in its type (`Is<T, "...">`), and outside text in `Untrusted<T>`. The runtime then
  checks and repairs them, with no instructions needed.
- Refining until good enough is `iterateOn`: ``iterateOn(revise, draft).until(nl`draft answers every question in
  message`)``. The step function returns the next state, and a natural-language stopping check is allowed.
- The same function under another policy is `fn.in(context)`; on a stronger model it is `fn.on({ model: NAME })`.
- A decision with both a code rule and a judgment uses `pluggable({ crisp, nl }, setting)`. In `shadow` mode both
  run, and their disagreements are recorded.
- Repeated inputs compile to code: `natlang specialize` turns a function's recorded calls into checked crisp cases
  that answer without the model, and hands everything else to the model.
- The program improves from its runs: `natlang tune` trains the local model on this program's own examples, and
  `natlang improve` searches for better instructions against them.

## The model

On exit code 3, ask the user before running `natlang setup --yes` (a one-time 1.8 GB download), or use a hosted
model: `--provider ID` (`natlang models`) or `NATLANG_SERVER=URL NATLANG_MODEL=ID`. `natlang doctor --json` reports
the setup.

The natlang-authoring skill (`natlang skills install`) has the full language: composition, transactions over
folders, Neuralese values, recursion and decisions.
```

The executor verifies every command and API in this text against the CLI and `@natlang/node` before shipping, and
drops or rewrites any line that does not hold. Example: if `natlang improve` still requires a `CASE.json` when F1
ships, that line says what it does today, until WS8 N4 lands. `createFunction`'s stub drops "Number the steps if
there are several" and becomes: "Say what to decide or produce from `text`. Keep exact steps in a TypeScript
helper this function calls."

## 4. F2: the authoring skill (3 days)

`SKILL.md` keeps its working loop, "A named function", "How a call executes" and "Common errors". It replaces the
opening and "Design the program":

**Opening (replaces L8-11).**
> Natlang is TypeScript with natural-language functions: typed functions whose bodies are instructions a model
> carries out, checked like any other code. Use it to automate repeat agentic work on a small local model: the
> program's types, data and exact steps are TypeScript, each judgment is a small typed function, and the runtime
> makes finite judgments calibrated decisions, compiles repeated paths to code and checks every result against its
> type.

**"Design the program" (replaces L88-121), eight items:**
1. **Write the types first.** Records and literal unions in `types.ts`, with `Untrusted<T>` on every value that comes
   from outside and `Is<T, "...">` on every property you would otherwise ask for in a sentence.
2. **Exact steps are TypeScript.** Arithmetic, parsing, sorting, joining, counting, record assembly, I/O: functions in
   the callable folder or the host. A `.nl` function calls them.
3. **Each judgment is one typed function.** Inline `nl<T>` in the code that needs it, or a `.nl` file when it is
   reused or tested on its own. When the instructions are several steps of judgment, number them and name the data
   each step uses. Children get distinct sub-questions and the inputs they need.
4. **Finite answers are decisions.** `readout: decision` on a literal-union or boolean result: one scored pass, and
   `decide` returns the probabilities. Gate actions on confidence, and escalate the unsure cases with
   `fn.on({ model })`.
5. **Compose.** Map a judgment over a collection with `Promise.all`. Pass semantic functions as arguments. Rebind a
   function's context with `fn.in(context)`. Use `pluggable({ crisp, nl })` where a rule and a judgment compete.
6. **Iterate structurally.** `iterateOn(step, state).until(check)`, where the check can be code or `nl`; recursion
   only on a smaller argument; `for...of` and array methods. There are no while loops.
7. **Change files and code through transactions.** `folder.propose(reducer)`, check `proposal.folder`, then
   `folder.accept(proposal)`. A verifier is a service, never a helper the model can edit.
8. **Close the loop.** `examples.json` holds the cases. `natlang specialize` compiles hot paths to crisp cases,
   `natlang tune` adapts the local model, and `natlang improve` searches instructions against the cases.

Deleted sentences: "Most applications are ordinary code with a few `nl` decisions where meaning matters"; "the author
(you) puts the algorithm ... and the interpreter only has to take the next step"; "Port an algorithm as its steps
over named data". The last one is replaced by item 3's narrower wording: numbered steps for judgment, TypeScript for
mechanism.

**New `references/composition.md`** (outline, ~6 KB): passing and returning semantic functions; `nl.with` vs
capture-by-mention; `.in(context)` and `Context.of`; `pluggable` modes and the shadow trace; `fn.on` and escalation;
`Folder` `propose`/`accept`/`snapshot`/`branch`/`select` and `folder.iterateOn`; `iterateOn` with a natural-language
check and a measure. Each section is one runnable example from `examples/` or a flagship (WS6), with no wrong form.
`references/patterns.md` is rewritten around `examples/triage`, `examples/review`, `applications/notebook`,
`applications/helpdesk`, `applications/nldb` and the flagships. Its Astra-derived paragraphs go (WS7 deletes or
migrates their source). `references/language.md` gains `Neuralese<T>` (WS1 L8) and recursion (`recursive:`, when L7
lands). The skill's error table gains every new code from WS1-WS3, each with its one-sentence fix.

`assets/review/` stays the bundled example. It already shows `Untrusted`, a decision leaf and an exact TS aggregate.

## 5. F4: README opening (exact text for the first screen)

```markdown
# natlang

natlang lets you automate work you would hand to an AI agent with a program you own, running on a small local
model. TypeScript holds the types, the data and every exact step; each judgment is a typed natural-language
function. Finite judgments are scored decisions with probabilities, outside text is fenced by its type, results are
checked against their declared properties, repeated paths compile to plain code, and the program improves from its
own runs.
```
followed by the existing `triage` snippet extended by three lines: `readout: decision` on `classify.nl`, a
confidence gate with `fn.on({ model })` for unsure tickets, and an `Is<string, ...>` summary type. Then "Get started"
as today, with "Your first program" walking through the F1 template instead of the classifier. The "Using natlang"
reading list leads with the curated programs (`examples/triage`, `examples/review`, `applications/notebook`,
`applications/helpdesk`, `applications/nldb`, then the WS6 flagships as they land).

## 6. F5: website (2 days)

Keep the nine-chapter design. Change content only:
- Hero: the F1 `main.ts` + `route.nl` pair replaces `classify.nl`/`triage.ts`. The `is_urgent.nl` code / prose /
  shadow exhibit stays (it already shows `pluggable`).
- Ch. III "Closures in English": add a `.in(context)` exhibit (one function, two policies).
- Ch. V "Termination": add `iterateOn(...).until(nl\`...\`)` and folder `propose`/`accept`.
- Ch. VII "A JIT for English": becomes the ladder: decision readout → crisp cases (`specialize`) → escalation →
  `tune` → `improve`.
- Ch. VIII "Neuralese": add `view` and a soft-state `semanticReduce` once WS8 N3 lands.
- Ch. IX "Policy & mechanism": rewrite around the eight design items of §4.
- App cards (`index.html:702-707`): remove "a treaty-obligation ledger" (WS7) and link the flagships.

## 7. F6: playground progression (3 days; examples land as their constructs do)

The gallery becomes an ordered progression; each example runs in the page (wllama) and in Node and is checked by
`natlang check` (and `--idiom` once I2 exists):

| # | Example | Construct | Available |
|---|---|---|---|
| 1 | Inbox planner (the F1 template) | decision readout, inline `nl`, `Untrusted`, `Is` | now |
| 2 | Calibrated gate | `decide` probabilities, confidence floor, `fn.on` | now |
| 3 | Map and tally | `Promise.all` over typed judgments + exact aggregation | now (rewrite of `support-triage`) |
| 4 | Refine until it holds | `iterateOn` with an `nl` stopping check | now |
| 5 | Two lenses | `.in(context)` | now (Wave-0 types) |
| 6 | Rule or judgment | `pluggable` shadow with disagreement trace | now (from `is_urgent`) |
| 7 | Safe edit | `folder.propose` → check → `accept` | now |
| 8 | Compiled cases | `specialize` on recorded calls; crisp hits shown | now |
| 9 | Semantic join and rank | `semanticMatch`, `semanticRank` | with WS3 S1 |
| 10 | Policy compiler | `compileRule` returns a lambda | with WS1 L4 + S3 |
| 11 | Soft summary | `view`, `semanticReduce(..., { carry: 'neuralese' })` | with WS8 N3 |
| 12 | Structured decision | record `readout: decision` | with WS2 R3 |

The 14 `natural.mjs` examples that use pre-v0.5 phrasing ("Write the number 7 to return") are rewritten into rows
1-8 or deleted. `interfaces.mjs`'s LLM-regenerated JSON DOM (`interfaces.mjs:37-53, 79-97`) becomes exact rendering
from a typed view plan (`ts-host/examples/browser-board` style).

## 8. F7: errors that teach the paradigm (with WS1-WS3)

Every check that detects a non-idiomatic form says the idiomatic one in one sentence. New messages, written out:
- `idiom-decision-readout`: "`{fn}` returns one of {n} values: add `readout: decision` to score them in one pass, or
  keep the tool loop if it must call helpers first."
- `idiom-fixed-cursor-iterate`: "This `iterateOn` steps through a fixed list one item at a time: map the step over
  the list with `Promise.all`, and keep `iterateOn` for steps that depend on the previous state."
- `idiom-code-in-prose`: "`{file}` describes code (`{token}`) in its instructions: write that step as a TypeScript
  helper in `{folder}/` and call it from the function."
- protected edit (WS1 L3): "`{path}` decides whether this call succeeded, so it cannot be edited here; fix the code
  that produces the value it rejects."
- recursion (WS1 L7): the existing "called itself without a smaller argument" message stays, and gains "declare
  `recursive: structural` to allow descent on part of an argument" when the definition lacks it.

## 9. F3: integration skill (1 day)
"Choose the smallest adequate boundary" becomes "Choose the boundary": the same five steps, plus one paragraph on
when the whole workflow should be a natlang program (it decides things repeatedly, and its results can be checked),
with `applications/helpdesk` as the example of a host application whose policies are natlang functions.

## 10. Acceptance for WS4
- F1, F2, F4 ship in Wave A. `natlang init` + `natlang check` + `natlang test` pass on local LFM3, and the first run
  is under ~60 s on DGX CPU.
- Blind authoring check, run once per front-door release: 5 fresh coding agents, each given only a new `natlang init`
  directory, the installed skills, and one prompt ("Build a natlang program that ..."; inbox, PR review, schema
  mapping, ticket routing, document QA). It passes when ≥ 4 of 5 programs have `.ts` exact steps, typed `.nl` or
  inline judgments, at least one `readout: decision`, and `Untrusted` on outside text, pass `natlang check`, and raise
  no idiom-floor warning. The 5 programs and their scores are committed to `ops/semantic-fp/front-door-checks/`.
