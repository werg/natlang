# Fused natural-language pipelines

Status: built up to the point where only a qualified model is missing (2026-10-09); follow-ups done the same day
(observed readers, explicit chains in the compilers and nldb, call-site marks for crisp TypeScript). Implements
[ARCHITECTURE_IMPROVEMENT.md](ARCHITECTURE_IMPROVEMENT.md) E2. Code: `ts-host/src/fusion/` (analysis, planners,
verifier, settings, report, observed readers), `ts-host/src/compiler/call-flow.ts` (the data-flow analysis the fact service
and the compiler share), `ts-host/src/runtime/fusion.ts` (hand-off at run time), `applications/fusion-planner/` (the
natural-language planner), `ts-host/scripts/fusion-bench.mjs` (measurement), `ts-host/test/fusion.test.mjs`.

## The idea

Today every hand-off between natural-language functions goes through text: stage A decodes its result to tokens, the
runtime parses and validates it, and stage B reads the tokens again. For a chain whose middle value nobody reads, that
decode and re-encode is waste, and it is lossy. If A and B run on the same model, A can leave its result as a Neuralese
block and B can read the block. Neuralese becomes the calling convention between functions, and the declared types do
not change.

## Owner decision: the fusing compiler is a natlang program

Policy is natural language; crisp code is mechanism. So the work is split in three, and the policy part is selectable
through `pluggable()` ([runtime/pluggable.ts](../ts-host/src/runtime/pluggable.ts)):

1. **Crisp fact service** (`fusion/facts.ts`, exact). For a project it returns the candidate edges of the call graph
   with the facts a planner needs: producer and consumer, the declared type, each side's model, and every reader of the
   value. It decides nothing.
2. **Natural-language planner** (`applications/fusion-planner`, an ordinary natlang program: `planner.nl`,
   `planner/decide.nl`, `planner/needsReading.nl`). Input: the facts. Output: one `EdgeDecision` per edge, `fuse` or
   `keep-text`, with a reason. The numbered steps of `decide.nl` spell out the fusibility conditions for a small
   model, and `needsReading.nl` is the one judgment ("keep text when the intermediate is something a person or a
   later check needs to read"), split out per the port-granularity rule: conditions are steps of the deciding
   function, only the judgment is a function of its own. Built-in programs live under `applications/` (the
   specializer's convention: an application directory located from the checkout, `NATLANG_FUSION_PLANNER`
   overrides).
3. **Crisp verifier** (`fusion/plan.ts`, `verifyPlan`). It checks any plan against the facts: no fused edge has an
   outside reader, both sides share model and dialect, the consumer parameter is known and its type is the
   producer's, and a reader set that was only observed has at least the configured number of supporting runs. A rejected plan goes back to the planner once with the problems; edges still rejected, and a planner
   that fails, fall back to text for those edges (`repairedPlan`). The natural-language planner can only make a plan
   more conservative than the facts allow, never less safe.

Selection: `fusion.planner` in natlang.json, `crisp` (default, `crispPlan`: the same rules plus "a finite result stays
text", no model needed), `nl`, or `shadow` (both run, crisp is served, `pluggable_shadow` records agreement).
`natlang check` reports the plan of the selected planner (`--fusion-planner` overrides for one run; `nl` and `shadow`
need a model, like `natlang call`).

## The fusibility rule

An edge is a producer function A whose result reaches a consumer function B, inside one orchestrator (a natural-language
function with stages in its callable folder, or crisp TypeScript that calls named functions). It is fusible exactly when:

1. **One reader.** Every reader of the value is B, and each is certain. Readers are classified as
   - `consumer`: the call B receives the value in (`B(A(x))`, or `v = A(x)` then `B(v)`);
   - `other-call`: another function receives it (fan-out);
   - `crisp-code`: crisp TypeScript, either a callable-folder module the value is passed to, or host code that touches
     the variable at all (a property access, a comparison, a log, a store);
   - `service`: a host service (`toolchain.verify(module)`, an index, a store) receives it, or something built from it;
   - `eval`: the orchestrating model reads it in its own code or reasoning: a field access `v.x`, an index, a condition
     (`when v is "ignore"`), a loop, a record it builds from it, or a store into another value;
   - `host-return`: the value, or part of what is returned, goes back to the host;
   - `trace-ui`: the prose logs, reports, shows or records it.
   A reader the analysis cannot see (a chain only implied by prose, or connected only by declared types) counts as an
   unknown `eval` reader, so the edge stays text.
2. **Same model.** Producer and consumer name the same model (`model:` frontmatter; none means the default).
3. **Same dialect.** One runtime dialect (`DefaultDialect`, which the certificate must match exactly).
4. **Not already soft**, and the consumer parameter that receives the value is known and has the producer's declared
   type exactly (any conversion between the two types is a computation, not a hand-off).
5. **A judgment, not a rule** (planner only): the intermediate is not something a person or a later check needs to read.
   The crisp planner approximates it by one rule: a finite result (`boolean`, a union of literals) costs about one
   token as text, so it stays text.

What "nothing crisp reads it" means:

- *Callable-folder TypeScript* (a `.ts` module beside the `.nl`, `gather.ts` in logs): any function that receives the
  value reads it. In prose, a call `module.fn(…v…)` is a `crisp-code` reader. In crisp TypeScript orchestrators the
  analysis is syntactic and exact: every other reference to the variable in its function is a reader.
- *Eval*: the orchestrating model writes eval code that holds the value. Passing the variable as an argument to the
  planned consumer is not a read. Everything else is, including string conversion and `read(v)`.

Fan-out is never fused: a value that two functions receive has two readers. (A later extension may fuse each leg; the
block is immutable and shareable, so the mechanics allow it, but the equivalence evidence is per hand-off.)

### Observations from the corpus

Prose orchestrators rarely say precisely who reads a value, so most candidate edges are not provable from the source. Counts
of candidate edges found by `fusionFacts` and the crisp plan, as `natlang check --fusion` reports them over the applications
of this checkout (fusion-planner excluded). "Before" is the first measurement; "now" is after the explicit chains below
(the first table gave migration nine edges, which was its scope count; the check lists six).

| Application | Orchestrator scopes | Candidate edges | Fusible before | Fusible now | Kept as text (why) |
| --- | --- | --- | --- | --- | --- |
| compilers | 1 | 6 | 0 | **3** | `parse -> analyze` in C, Python and Rust now fuse; `analyze -> declare` (3) is read by the orchestrator (`checked.diagnostics`) |
| nldb | 4 | 8 | 0 | **1** | `parse -> plan` fuses; `plan -> execute` is read for its columns and explanation; the executor's 6 operator hand-offs follow whatever plan the optimizer wrote, so no static chain exists |
| pi | 18 | 4 | 0 | 0 | 3 fan-out (`request` goes to classify, answer, startToolRound), 1 implicit with field reads |
| wiki | 5 | 2 | 0 | 0 | 2 typed-only |
| logs | 3 | 3 | 1 | 1 | `runbook -> hypothesize` fuses; `hypothesize -> queries` is stored into `inc.hypotheses`; `escalate -> summarize` has field reads and 5 more readers |
| build | 2 | 3 | 0 | 0 | field reads (`graph.…`), host return |
| migration | 9 | 6 | 0 | 0 | fan-out of `understand` to five stages; `plan -> summarize` field reads; typed-only |
| scheduling | 1 | 3 | 0 | 0 | `order` is crisp code; field reads |
| workflow | 3 | 3 | 0 | 0 | values are logged and shown; fan-out |
| games | 4 | 0 | 0 | 0 | no stage-to-stage hand-offs in prose |
| Total | | 38 | **1** | **5** | |

No crisp TypeScript orchestrator in these applications chains two natural-language calls; their hosts call one function
per request. The test project (`fusion.test.mjs`) covers the TypeScript analysis, the call-site marks and the run time.

Reading the result: the rule is sound but the corpus was written for text. Two ways out of the typed-only edges, both
without changing a declared type, and both done:

1. **An author writes the chain explicitly.** The language already has the form: a call inside a call
   (`checked = analyze(parse(source))`), or a name used only as the next stage's argument. The analysis proves both. No new
   syntax was added; [references/fusion.md](../skills/natlang-authoring/references/fusion.md) of the authoring skill says how
   to write a chain and how to make the next stage carry what the orchestrator needs from the middle value (the
   compilers' `analyze` now starts its diagnostics with the parser's, so the orchestrator reads only `checked`). Faithfulness
   is the author's call, edge by edge: where the orchestrator needs the middle value or a plan decides the chain at run time
   (nldb's executor), the edge stays text.
2. **Observed readers** (next section).

## Observed readers

The fact service can read readers from the eval code orchestrating models actually ran, as the call store recorded it
(`approach.evals` of the orchestrator's calls). That code is exact TypeScript, so its readers are exact.

- Switch: `fusion.observed` in natlang.json (`true`, or `{ "minRuns": 20, "store": "path" }`); absent means off, and
  `fusionFacts(root, files)` alone never looks at a store. `natlang check`, `natlang run` and `analyzeFusion` pass the machine
  store (`NATLANG_CALL_STORE`) unless `store` names another.
- Which runs: calls of the orchestrator at its source path, executor `agent`, with `isModelEvidence` (a declared model that
  spent tokens; scripted drivers do not count), whose recorded instructions equal the instructions now (an edit starts the
  count again). The newest 500 are read.
- What is observed: for edges whose only unknown is the prose (`typed` and `implicit` flows), each run's eval programs are
  analyzed as one program with the same data-flow analysis the crisp TypeScript scopes use. A run **supports** an edge when
  every call of the producer in it passes its value to the consumer's parameter and to nothing else (a name used only as that
  argument, or a nested call); it **contradicts** when some call is read by anything else: a field access, a condition, a
  service or `console.log`, the orchestrator's result, a destructuring, a second consumer, a producer whose value is never
  handed on. Variables span the evals of one run. The reader set is replaced by the observed one only for edges with no
  certain outside reader already.
- Evidence: the edge carries `observed: { runs, contradicted, minRuns, revision }`; the revision is the number of runs read
  and a digest of their call IDs. The plan records `evidence` for every observed edge (readers: observed, runs, contradicted,
  minimum, revision), `natlang check --fusion` prints `[readers observed: N runs, store R]`, and the planner program sees
  `edge.observed` and keeps text below the minimum.
- Verifier: `unfusableBecause`, `crispPlan` and `verifyPlan` take `observedMinRuns` (default 20, from `fusion.observed.minRuns`).
  An observed edge with fewer supporting runs is kept as text with the count in the reason; a contradicting run lists its
  readers on the edge, so the edge is not fusible whatever the count. An NL planner that fuses it anyway is rejected like any
  other bad plan.
- Observation is statistical, unlike a proof from source: a later run may read the value, and a block then reaches code that
  expects text. Use `shadow` for observed edges before `on`, and treat a contradicting run as a reason to rewrite the prose.
  On this machine's store the model-driven runs of the repository's orchestrators number zero (its records are scripted), so
  the counts above are from source alone.

## Call sites in crisp TypeScript

Hand-offs in crisp TypeScript (`analyze(await parse(source))`, or a name used only as `analyze(tree)`) are analyzed exactly
(the shared analysis in `compiler/call-flow.ts`) and now run fused. The soundness argument:

- The analysis is syntax over the enclosing function (the whole file for a top-level name). A name counts as read at every
  mention that is not exactly an argument of the consumer, including `export`; a shadowing name only adds readers. Nothing
  sits between the two calls, so no TypeScript code sees the block.
- The project build (`buildProject`, and the same lowering as runtime module loading) wraps each call of a candidate hand-off
  as `__natlang.fuseSite(id, (...a) => f(...a), ...args)`. The arguments are evaluated where they stand. With fusion off the
  wrapper calls the function and nothing else. The mark is the call's file and text offset (`file@offset`), the same value
  the facts carry in `edge.sites`.
- The plan is made at run time (settings, certificate); the mark is only a place. `fusedEdges` returns a TypeScript edge with
  its `sites`; `engageFusion` matches the call by the frame's site and the edge's function, retypes the producer's result or
  the consumer's parameter as for any edge, and falls back to text with the same events when the certificate does not hold.
  Events of calls the host made have no calling function's trace; `fusionHostEvents(task)` lists them.
- Unchanged risk: a block left unconsumed (the consumer is on a branch that did not run) is reported by
  `unconsumedFusedBlocks`. Calls in code that never went through the project build are not marked, so they stay text.

## How the type changes

Declared types stay `T` everywhere an author sees them: signatures, `.d.nl.ts`, the declarations in the model's eval
scope, `natlang check` diagnostics. Fusion is a compilation decision invisible in source. At the hand-off the runtime
changes the effective types of one call pair:

| Where | Declared | Effective on a fused edge |
| --- | --- | --- |
| Producer A's result | `T` | `Neuralese<T>`: A answers as it would for a declared `Neuralese<T>` result. The runtime writes the block through the existing typed-result path (soft readout). |
| Consumer B's parameter | `T` | `Neuralese<T>`, when the argument is a fused block. B sees it as a soft argument. A text argument (the producer fell back) leaves B as written. |
| The orchestrator's eval binding of A's result | `T` | the block is accepted where `T` is declared, for blocks this task's fused hand-offs wrote (`native/fusion-blocks.ts`, checked in `coerce`). |

The mechanism is `engageFusion` in `runtime/fusion.ts`, called once from the kernel (`runDefinitionBody`) before the
call's signature is fixed, plus one line after the value is known. It finds the orchestrator from the parent call's
trace manifest, matches the call against the planned edges of that scope, retypes the producer's result or the
consumer's parameter, and emits trace events. Everything else is existing soft-value machinery: store, port, typed
result writes, soft argument rendering.

Scope of the runtime: fused hand-offs inside natural-language orchestrators (a call's parent is a natural-language call,
so the runtime sees both sides), and inside crisp TypeScript orchestrators whose calls the project build marked (see "Call
sites in crisp TypeScript").

## Qualification: when fusion may turn on

Fusion is only as good as the channel. `on` engages only when all of these hold (`fusionStatus`):

- the runtime has a Neuralese store and write port, and the model driver declares Neuralese capability;
- the model profile carries a **runtime-qualification certificate for the exact weights and dialect**:
  `natlang.fusion-certificate/1`, issued by the training pipeline and named by `fusion.certificate` in natlang.json
  (the profile's weights digest is `fusion.weights`);
- the certificate records passed runtime qualification (TRAINING_RECIPE.md stage 4: the production encode, read and
  write path, token boundaries, gradient replay, typed task execution; stage 5 for typed function execution) and
  the passed self-feedback gates of DECISIONS.md 2026-10-09 (channel equivalence, generation quality);
- model id, weights digest and dialect match the running profile and port exactly.

Anything else falls back to text for every planned edge and traces `fusion_fallback` with the reason, once per edge and
task. Legacy marker/RMS checkpoints and the foundation certificate do not satisfy it. A backbone change is a new
certificate.

The text emulation (`model/text-neuralese-emulation.ts`) exercises the whole path in tests. A host declares it with
`fusion.emulation = { dialect }`; it is accepted only when the port writes exactly that dialect, and every fusion trace
event then carries `emulation: true`. It is not a certificate: the benchmark gate fails for an emulated run.

### What the training side must deliver

One JSON file per (model, weights, dialect), written after stage 4/5 qualification of those weights, never inferred
from a checkpoint's inheritance:

```json
{
  "schema": "natlang.fusion-certificate/1",
  "model": { "id": "<profile model id>", "weights_sha256": "<digest of the served weights>" },
  "dialect": "nd:<name>@<version>",
  "runtime_qualification": {
    "status": "passed",
    "scope": ["transport", "gradient-replay", "typed-function-execution"],
    "report_sha256": "<digest of the runtime qualification report>",
    "self_feedback": { "channel_equivalence": "passed", "generation_quality": "passed" }
  },
  "foundation_certificate_sha256": "<optional>",
  "issued_at": "<ISO time>", "issuer": "<pipeline>"
}
```

and a server speaking that dialect (`GET /v1/neuralese/info`, `neuralese: true` driver, `neuraleseServerModelTurn`) that
can write a block as a typed function result and read one as a soft argument in the same request. The runtime
profile then sets `"fusion": { "mode": "shadow", "certificate": "cert.json", "weights": "<same digest>" }`.

## Modes

`fusion.mode` in natlang.json, default `off`:

| Mode | Behaviour |
| --- | --- |
| `off` | Nothing changes, nothing is traced. The runtime option is not even installed. |
| `shadow` | Text is served. For each planned producer call a second call of the same producer runs with a soft result (traced `fusion_shadow`, `scope: producer`): its block is read back and compared with the text value (`agree`, `shadow_ms`). The consumer is not re-run, because it may have effects; a host that knows it is pure may extend the comparison. End-to-end agreement comes from the benchmark, which runs the chains in a sandbox. |
| `on` | Planned edges are fused where the certificate holds; fallback otherwise. |

Trace events, on the orchestrator's call: `fusion_edge` (`status`: `fused` | `consumed` | `fallback`), `fusion_fallback`
(`reason`), `fusion_shadow` (`agree`, `block`, `error`, `shadow_ms`). The planner's own comparison appears as
`pluggable_shadow` (name `fusion-plan`).

## Measurements and the gate

`ts-host/scripts/fusion-bench.mjs` runs one application's entry over a case file as text and fused, against the same
Neuralese-capable server, and reports for each mode: tokens (prompt, completion), turns, latency (total, p50, p95),
errors, accuracy against `expected` when cases carry it; and for the pair: output **agreement** (canonical equality case
by case), token ratio, latency ratio, whether fusion engaged, the fallback reasons and the count of fused blocks nobody
consumed.

```sh
cd ts-host && node scripts/fusion-bench.mjs --app ../applications/compilers --entry compiler.nl \
  --cases cases.jsonl --neuralese-endpoint http://HOST:PORT --model MODEL \
  --certificate cert.json --weights SHA256 [--planner crisp] [--out report.json]
node scripts/fusion-bench.mjs --app ../applications/compilers --entry compiler.nl --cases cases.jsonl --dry-run
```

**Gate for enabling fusion for an application** (all of): the text pipeline runs the cases (no errors); fusion engaged
(at least one fused edge and none fell back); output agreement at least 0.98 (and no loss of accuracy where `expected`
exists); total tokens of the fused run not above the text run; total latency not above; every fused block consumed;
not an emulated run. Then `shadow` on live traffic must show `fusion_shadow.agree` at the same level before `on`. The
defaults are in `DEFAULT_GATE` and are meant to be tightened by evidence, not loosened to pass.

The benchmark was not run against a real model; the test suite runs it on the text emulation with scripted models only
to prove the harness and that an emulated run can never pass the gate.

## Tests

`ts-host/test/fusion.test.mjs`: detection on small projects (bound, nested, service, different models, TypeScript
orchestrators); the verifier rejects a bad plan, the retry carries the problems, ignored problems fall back to text, a
failing planner falls back; the natural-language planner program runs on a scripted model and its verified plan equals
the crisp one; fusion off changes nothing; fused end to end under the emulation (the consumer is shown a block, traces
mark the emulation); fallback without certificate, without Neuralese backend, with a mismatched certificate;
shadow agreement and disagreement; `natlang check` text and `--json`; the benchmark harness. Observed readers: supporting and
contradicting runs, the minimum in the crisp planner and the verifier, the evidence in plan and report, reads the analysis
must see (field, return, display, fan-out, destructuring, evals sharing a name), the call store adapter (model evidence only,
same instructions, revision), the settings. Explicit chains: nested and bound forms proven in prose, a field read not. TypeScript
call sites: the facts' site IDs, an exported name, a project built with the marks, and the emulated run (the planned
hand-offs pass a block, a re-read or shared value and fusion off stay text).

## Open items

- Observed facts have no real input yet: the repository's own store holds scripted runs only. They need live orchestrator runs
  (the teacher window) before any edge can pass `minRuns` on evidence.
- More explicit chains: wiki and migration still have typed-only edges; the compilers' `analyze -> declare` would fuse if
  `declare` carried the checked diagnostics forward (`declare(analyze(parse(source)))`); that moves the stop condition into
  `declare`, an author's decision when the compilers are measured.
- Consumer-side shadow for consumers declared pure.
- A real certificate, a qualified server and a benchmark run (training side).
