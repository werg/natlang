# Hosts and model drivers

## One runtime, ordinary calls

On a machine with natlang model profiles, take the runtime the CLI uses:

```ts
import { natlangRuntime } from '@natlang/node';
import { handle } from './app.js';

const natlang = await natlangRuntime({ programRoot: import.meta.dirname, runtime: { services: { wiki } } });
try { const report = await natlang.runtime.run(() => handle(ticket)); }
finally { await natlang.close(); }                 // ends model sessions and the servers it started
```

It selects the profile (`profile`, `NATLANG_PROFILE` or the default), runs a tuned program's active release on the
checkpoint it was tuned on and loads the release into the server, starts a native profile's server (and stops it in
`close()`), wires the profile's `escalate` teacher through the tiered engine, and writes one-line notices (tuned
binding, newer checkpoints, escalations) to `log`. An escalated root call's input joins the program's
`tune.examples.jsonl` when the runtime closes. `environment` replaces `process.env` for profiles and the call store.
The result also has `driver` (the selected model, for direct calls), `selection`, `executorIdentity`, `tuned` and
`prepare()`. A server natlang started without a launcher also stops when the process exits.

A native server holds its model (LFM2.5-350M: about 2.4 GiB host memory with the actor view), a prompt-prefix cache
(`native.prefixCacheGb`, default 2) and, for tuned programs, each adapter stack's effective weights (about 1 GiB per
stack for LFM2.5; `NATLANG_ADAPTER_CACHE_GB`, default 4). On a shared machine, launch it through the profile's
`native.launcher` (a memory ledger unit) and keep one server per checkpoint: every process finds it on the profile's
port, and `natlang serve` keeps it loaded between commands.

To wire the model yourself, build the runtime directly:

| Target | Package | Platform | Context propagation |
|---|---|---|---|
| Node | `@natlang/node` | `nodePlatform()` | `AsyncLocalStorage` |
| Browser | `@natlang/browser` | `browserPlatform()` | Compiled code restores the task after each `await`; `runtime.bind` for uncompiled callbacks |
| workerd | the core bundle (`natlang-core.js`) | `workerdPlatform({ realm })` | `AsyncLocalStorage` (`nodejs_compat`) |

What a runtime runs on is a `Platform` value passed to `createNatlangRuntime({ platform })`: the context store, the evaluator for a task's evals (`environment(options)`) and for callable-folder modules (`moduleRealm()`), the default call store, platform modules (`builtinModules`: `natlang:neuralese` ...), the package loader, the `bash` and `python` folder tools, and the `return_result` wording. Nothing is registered process-wide, so runtimes on different platforms coexist in one process and a platform's module instances never serve another's. `@natlang/node` and `@natlang/browser` pass their own platform when you pass none (`createNatlangRuntime`, `compileVirtualProject`, `loadVirtualNatlang`, `loadVirtualCallables`, `projectEntry` take an optional `platform`); the core's `createNatlangRuntime` requires one and fails with `NatlangPlatformUnavailableError` otherwise. Make a variant with a spread (`{ ...nodePlatform(), builtinModules }`); module state belongs to the platform value. The Node platform reads `NATLANG_CALL_STORE`, `NATLANG_SPECIALIZATION` and `NATLANG_RETURN_ERROR_TEXT`; the kernel (`runtime/`, `native/`, `calls/`, `compiler/`) reads no environment and no home directory (`machineStoreRoot`/`openMachineStore` live in `machine-store.ts`, the CLI passes `--group`/`NATLANG_FEEDBACK_GROUP` itself).

An evaluator is the shared one over a `Realm`: `realm.call(body, parameters, args, { timeoutMs? })` runs a strict-mode function body with the named parameters in the platform's global scope and returns its result. TypeScript compilation (`compileEval`), the scope snapshot, the console and the failure reports are shared. Node's realm is a `node:vm` context, the browser's the page's Function constructor, and a host's anything it can build: `constructorRealm(newFunction)` wraps `(body, name, ...parameterNames) => fn`, which is workerd's `UNSAFE_EVAL.newFunction`. `realmPlatform({ name, contextStore, realm })` makes a platform from a realm. Compiled projects (`compileVirtualProject`) and callable-folder modules run through the platform's module realm too, so a workerd host never needs `new Function`.

Compiled program code (the output of `compileVirtualProject`) reaches the runtime only through `require(specifier)` for the project's `runtimeSpecifiers` (default `@natlang/browser`, `@natlang/node`) and `NEUTRAL_RUNTIME_SPECIFIER`; the loader answers all of them with the `runtime` namespace passed to `compileVirtualProject`. A host that loads the core once per isolate passes that one module namespace, so every compiled program binds to the same instance, the same context store and the same `nl`/`iterateOn`. The core also exports the host-cell compiler (`lowerHostCell`, `declarationsForValues`) and Pi's pure request/reply translation.

```ts
import { createNatlangRuntime, fileTraceSink, openAICompatibleModelTurn } from '@natlang/node';
import { handle } from './app.js';               // compiled with natlang build

const runtime = createNatlangRuntime({
  model: openAICompatibleModelTurn({ endpoint: 'http://127.0.0.1:8080', model: 'natlang' }),
  services: { wiki },                            // host capabilities, see below
  trace: fileTraceSink('.natlang/traces'),       // optional: one JSONL per invocation
});
const report = await runtime.run(() => handle(ticket));
```

`models: { small: driver }` names further models: a named function whose frontmatter says `model: small` runs on that one, and on `model` when no entry has its name.

`runtime.decide(fn, ...args)` runs a `readout: decision` function in a task of its own and returns `{ value, probabilities: [{ value, probability }], confidence, scored }`, so the host can act on probability floors; a driver without a scorer gives the sampled value with probability 1 and `scored: false`.

`runtime.run(fn, { services, signal, trace, onLive, name })` creates a task: the natlang calls made anywhere inside `fn` (including in libraries and callbacks) find it. Tasks run concurrently. A natlang call with no task fails with an error naming `runtime.run` and `runtime.bind`. Model options: `model` may be a driver function or `{ driver, maxTurns, maxTokens, turnTokens, temperature, maxFailureRepairs, … }` (all optional; none is set by default); `limits`, `seed`, `workspace`, `network`, `statistics`, and `progressJudge` are runtime options.

To show a natlang function working, subscribe to live events: `onLive` on the runtime (every task) or per task. A `model_delta` event arrives for each piece of a call's model turn as it streams, `{ type: 'model_delta', taskId, callId, parentCallId, definitionId, name, turn, delta }`; `delta` is the driver's `ModelTurnDelta` (`text`, `reasoning`, a `tool_call` fragment, a written `neuralese` block, or `reset`). Show by call and turn (`callId`, `turn`; `parentCallId` nests a call under its caller); on `reset`, discard what that turn showed, since its request is sent again or failed. Live events are hints: the trace (`turn` is its `model_request` events' turn) and results are authoritative and do not change. The subscriber runs synchronously and must be quick; what it throws is ignored. Deltas come only from drivers that stream (`openAICompatibleModelTurn` does by default, the Neuralese driver when its server declares streaming); a driver receives the turn options (`(request, signal, { onDelta })`) only while someone subscribes, so a wrapping driver forwards its third argument. `call_start` and `call_end` (`outcome`, `detail`) bracket every call, and a `trace` event (`event`: the trace event, such as `model_request` with `prompt_tokens`, `action`, `scope_failure` or `model_incomplete`) arrives as each is recorded; `natlang run` uses these for its progress line and `natlang dashboard` for its live view. Without a subscriber nothing changes.

## Compile the application

`natlang build [PROJECT]` type-checks the project, plans every `nl` expression, checks callable folders, generates `foo.d.nl.ts` for each `.nl` file, and emits JavaScript. `natlang check` does the same without emitting. `buildProject({ project, runtimeModule })` and `checkProject` are the programmatic forms; `compileVirtualProject({ files }, runtimeNamespace)` compiles an in-memory project (browser pages, workers, tests). Uncompiled `nl` calls fail.

Named functions can also be loaded directly: `loadNatlang('review.nl')` (Node), `loadVirtualNatlang(files, 'review.nl')`, or `defineNatlang(nlSourceText)` for functions authored at run time (notebook cells, generated tools). `loadCallables('natlang.d')` loads a callable folder as a record. Tests that run application code importing `.nl` files under Vite or Vitest add `plugins: [natlangVitePlugin()]` (from `@natlang/node`): each `.nl` import loads as `loadNatlang` would, with the nearest `package.json` above it as the package root.

Inline `nl<T>` result annotations written in eval are checked against the eval scope's declared type names. A type alias shown only in an earlier natural-language instruction is not in that scope; `nl-undeclared-type` identifies it before the child silently becomes `unknown`. Declare the alias in scope or write the required structure in the annotation. A program alias named like a platform class (`Request`, `Response`, `URL`, `Date`, `Map`, ...) shadows the class in that program's scope, as in TypeScript; a host that wants the live built-in passes it under a type that is not redeclared. Before 2026-10-12 such an alias silently resolved to the platform class and every call mismatched (`type-mismatch, expected Request`); on an older runtime, rename the alias.

## Host eval cells

A host whose agents run source cells (TSX, top-level `await`, `import`, a top-level `return`) can make `nl` ambient in every cell, with no import. Natlang plans and lowers the cell before the host's own transform (sucrase, esbuild) runs:

```ts
import { lowerHostCell, declarationsForValues } from '@natlang/core';   // compiler/host-cell.ts

const ambientDeclarations = declarationsForValues({ scope: scope.snapshot(), threshold });   // `.d.ts` text; add the host's own API declarations
const { code, plans, diagnostics, usesNl } = lowerHostCell(cellSource, { ambientDeclarations, jsx: true, services, context });
if (diagnostics.length) return reportToAgent(diagnostics);   // 1-based line and column in the cell as written
// run `code` through the host transform; `nl` is the runtime's nl, which provides `nl.__inline`
```

- **Fast path.** A cell that never writes `nl`, `iterateOn`, `Neuralese` or a call of `read` returns at once, unchanged, with `usesNl: false`; no TypeScript program is built, so ordinary cells never pay for the checker. A host whose scope holds Neuralese values passes `neuralese: true`, which analyzes every cell (a value from the ambient declarations is coerced to text with no word of natlang in the cell). A cell that writes it but only as a property, a string, or a local declaration (a parameter, a `const nl = ...`) also returns `usesNl: false` and is not lowered. `nl` resolves by symbol: a local declaration shadows the ambient one, and is not an error here (it is in a project).
- **One implementation.** Inference is the project compiler's (`analyzeInlineLambdas`): explicit `nl<T>`, contextual type, immediate call arguments, result context, typed holes; conflicting evidence, `nl<T>` annotations the code needs, tags nobody calls and run-time-built instructions are reported with the teaching messages and the natlang codes. The lowered form is the one project builds emit: `nl.__inline(plan, values, accessors, context)`, with live getter/setter accessors for captured `let`s, so a `let` the child assigns is written back to the cell. ``await nl(`...`)`` and `nl<T>("...")` are the one-shot call, as in eval.
- **Output is still the host's language.** Types and JSX are untouched, each lowered site keeps the newlines of the code it replaces (later lines stay on their line), and a cell with an error diagnostic is returned as written. Lowering is the project transformer's, so everything it lowers in a project it lowers in a cell: `nl` sites, `iterateOn` call sites (wrapped with `withCompilerSite`, site `fileName#enclosing name`), typed Neuralese readouts (string coercion, `read(x)`, `join`, `JSON.stringify`, ...), generic-result instantiations and `refine`/`assume` predicates. The lowered code reaches the runtime support functions through the binding `__natlang` (`import { __natlang } from '@natlang/node'`); the host binds it in the cell's scope beside `nl`. Output positions are exact outside lowered expressions; inside one the code is printed from the syntax tree, so its internal formatting changes.
- **Ambient names are explicit authority.** `ambientDeclarations` is a global `.d.ts` (no top-level `import`/`export`) that types the host's free variables; a name nothing declares, and every imported module, is open (`unknown`/`any`). Declared names are not captured when an instruction merely mentions them: pass them as an argument or list them in `nl.with({ scope })`. Captures of the cell's own variables work as in any natlang file (mention captures live; `nl.with` snapshots, `live(x)` for write-back). A snapshot expression in `nl.with({ rows: await db.q() })` is evaluated in the cell, where `await` is legal.
- **`declarationsForValues(values, limits?)`** writes `declare const name: Type;` from the current values, by the types natlang gives scope values (records by field, lists by item union, large keyed objects as `Record<string, T>`), read to bounded items, fields and depth, cycle-safe. Functions are `(...args: any[]) => any`, built-ins (`Date`, `Map`) are named, other live objects are `any`, and a record that holds some keeps the types of its other fields (`inferValueType(value, { liveFields: true })`).
- **`services`** declares typed host services (the declarations of a program descriptor) in the same ambient scope; `context` is the callable folder (`natlang.d`) every `nl` in the cell is bound to, embedded in each site as in project builds; `libs` supplies TypeScript library declarations where TypeScript has no install on disk (workerd).
- **Execution.** Lowered code needs only an `nl` whose `__inline` is the runtime's (`import { nl } from '@natlang/node'`, or the host's provider); run it inside `runtime.run(...)`. The plans in the result are the ones embedded in the code, for hosts that list or index sites.

## Crisp and natural-language implementations of one part

A hot part with a crisp and a natural-language implementation (validation, settlement, a scheduler's policy, an entry function the host selects) is one `pluggable` call behind a setting, not a hand-written `if`:

```ts
import { pluggable } from '@natlang/node';
const settle = pluggable({ crisp: () => reference(state, ordered), nl: () => policy(state, ordered, problem) }, settings.settle, { name: 'economy.settle' });
return settle();
```

`pluggable({ crisp, nl }, mode, { name, serve, same })` returns an async function of the shared arguments. Modes: `crisp` and `nl` run one side; `shadow` runs both concurrently on the same arguments, returns the `serve` side (default `nl`, the policy of record) and records a `pluggable_shadow` trace event (`name`, `served`, `crisp`, `nl`, `agree`), the way refinement shadow mode records `refinement_shadow`. Results agree when their canonical values are equal; pass `same(crisp, nl)` to compare only what matters (a verdict's `ok` and not its wording). A failure of the side that is not served is recorded as `agree: false` and does not fail the call; the served side's failure does. Read the events as bug reports for the crisp side and hard negatives for the natural-language one; switch the setting to `crisp` when the disagreement rate is acceptable. It is a host and application helper: model-facing instructions never name it. `applications/games` (`games/economy/settle.ts`, the validators) is the example; the `Settings` type there takes `'nl' | 'crisp' | 'shadow'`.

This is the one mechanism for "crisp or natural language": do not write a `settings.x === 'crisp'` ternary, a hand-made dispatcher or a per-service `implementation()` branch. One vocabulary: `crisp`, `nl`, `shadow`. `{ default: 'crisp' }` sets the mode for an absent setting (default `nl`). `pluggableMode(setting, fallback)` normalizes a setting without building the function. **Deprecated spellings:** `natlang` (wiki and logs settings) and `natural-language` (the build, migration, scheduling and workflow `implementation` values and policy options) are accepted wherever a mode is and mean `nl`; write `nl` in new settings and configs. Settings that appear in the signature of a `.nl` stage (`WikiSettings`, `LogSettings`) keep their old type text until the change has been measured live, because that text is model-facing; the TypeScript side accepts both. A service's `implementation(point)` is only the way the setting reaches the stage; the stage still selects through `pluggable(…, await service.implementation(point), …)`. The three modes are accepted by those services at run time, but their declaration text (shown to the stages) still says `'crisp' | 'natural-language'`.

## Built-in natlang programs

Natural-language programs that ship with the runtime (the progress judge of `iterateOn`, the stopping-condition note, the behavior-comparison judge of the call store, `view` (the summarizer, whose body is the prompt piece `view`), the game policy) are `.nl` files in `ts-host/src/builtin/`, not strings in TypeScript. `scripts/generate-builtin.mjs` embeds them as `sources.generated.ts` (run by `build:node`, checked by `npm run check`), so the same bytes reach the Node build, the npm package and the browser bundle. They load as a callable folder (same parser, type checks and policy). Runtime code resolves one by name: `builtin('progressJudge')` is the callable (`builtinNames()` lists them); `builtinDefinition(name)` is the kernel definition for the runtime's own `invokeDefinition` calls; `builtinBody` and `builtinSource` (from `src/builtin`) give the text for prompt pieces. Prompt-piece ids are unchanged, and `test/builtin-programs.test.mjs` compares the rendered prompts with a snapshot taken before the move. Edit the `.nl` file, not the generated module. A change to a built-in's text is model-facing: measure it live and log it in `plans/MODEL_FACING_CHANGES.md`. The interpreter's own prompts (`TOOLS_PROMPT` and the like) stay in TypeScript: the interpreter cannot interpret itself. Put a new judge or policy of the runtime here first.

## Services

```ts
// app types (once):
declare module 'natlang:services' { export const wiki: WikiWorkspace; }
// callable-folder code:
import { wiki } from 'natlang:services';
```

Services are ordinary objects with methods, supplied per runtime or per task. Callable-folder TypeScript imports them from `natlang:services`; in eval they are named bindings. Every method call is traced as an effect. Services are read-only bindings: the model calls methods, it does not reassign properties.

Anything that stands for the world outside the program (stores, boards, simulators, verifiers, APIs) should be a service rather than callable-folder code: the model can read and edit callable-folder code, and it will edit a system that refuses it. Tell the model what a service is with its declaration, and limit who may use it where that matters:

```ts
const runtime = createNatlangRuntime({ model, services: { records, tables },
  // What the model is shown and read_code returns: a .d.ts body, wrapped as `declare namespace records { … }`.
  serviceDeclarations: { records: '/** Look a question up in the records. */\nexport function find(question: string): string;' },
  // tables is usable only in calls of answer/table_expert.nl and the calls they make; others see who can use it.
  serviceScopes: { tables: ['answer/table_expert.nl'] } });
```

Arguments reach a service as host-realm plain data, never as eval-realm objects, so a service may persist or compare them directly. A method that is an external effect to perform once (a provider request, a payment, a sent message) is listed under `ONCE_EFFECTS` (exported by `@natlang/node`): `{ [ONCE_EFFECTS]: ['send'], send(...) {...} }`. A repeated call with the same arguments on the same service object, from a re-run eval or another call, returns the earlier result; a failed call may run again. Hand a fresh service object to each unit of work (a task phase, a request) that may repeat the effect on purpose, and never list reads.

Both options also exist per task (`runtime.run(fn, { services, serviceDeclarations, serviceScopes })`). A service without a declaration is listed by its method names only. Scopes name functions by their source path relative to the loaded program. Importable packages need no declaration: `read_code("pkg")` reads a package's exports from its own type declarations.

## Model transport

A driver receives `{ messages, tools, temperature, seed, max_tokens }` and returns ordered tool calls plus optional text and usage:

```ts
{ calls: [['eval', { code: 'return await helper(sample)' }]], text: '', completion_tokens: 42, prompt_tokens: 700 }
// a later turn with no calls, e.g. { text: 'done' }, returns the staged value
```

A driver is an ordinary function, so it composes: wrap one to limit requests in flight, to route some calls elsewhere, or to answer a known call itself (the first pi app answered its codemode call with an `eval` of the agent's script); wrap it with `wrapDriver(driver, { turn?, decide? })` (`@natlang/node/model`), which copies the wrapped driver's properties so its decision scorer, `guidance`, `neuraleseRuntime` and `contextWindow` stay reachable (a hand-written `(request, signal) => driver(request, signal)` drops them, and every `readout: decision` call then runs the full tool loop). `openAICompatibleModelTurn({ endpoint, model, concurrency })` caps requests in flight over its turns and its decision scorer together (pass one `requestLimit(n)` to several drivers to share a cap); profiles take `concurrency` too, or `NATLANG_CONCURRENCY` per process. Requests wait in one model scheduler per backend (`createScheduler`, `model/scheduler.ts`): it sends up to `maxConcurrent` at once, releases requests that are ready together as one batch, serves turns of calls already running before new calls, sends requests with the same system prompt next to each other, and cancels queued requests by signal. Pass it as `concurrency` (a bare number keeps the plain FIFO `requestLimit`). Profile knobs: `batching: { mode: 'server-continuous' | 'explicit-batch' | 'serial', maxConcurrent, coalesceMs, priority, scoreEndpoint }` (a managed local server defaults `maxConcurrent` to its slot count; `NATLANG_CONCURRENCY` still overrides `concurrency`), and for a managed local server `local: { parallel, contextTokens, memoryBudgetMiB, kvBytesPerToken }`: unset `parallel` is sized from memory, default KV budget min(free/2, 4 GiB) (`NATLANG_MODEL_MEMORY_BUDGET_MIB`, `NATLANG_MODEL_KV_BYTES_PER_TOKEN`; formula in `llama/slots.ts`), `contextTokens` is per slot, and requests carry `cache_prompt: true`. `natlang doctor` reports `serverSlots` and `batching`. Decision scorers expose `scoreMany(items)`; decision readouts that arrive together are scored together, and with `mode: 'explicit-batch'` plus `scoreEndpoint` through one prefix-plus-continuations request (plans/BATCHED_EXECUTION.md). Each model request's trace event records `batch_id`, `batch_size`, `in_flight` and `queue_wait_ms`; `natlang traces occupancy` summarises them. `natlang check` warns about a `for…of` that awaits one independent nl call per item (`nl-sequential-loop`) and suggests `Promise.all(items.map(...))`; loops are never parallelized silently. `openAICompatibleModelTurn` adapts an OpenAI-compatible server (aliases, retries of malformed tool calls, raw exchanges via `onExchange`). `createManagedModelSession` and `natlang setup` run the managed local llama.cpp runtime. Keep provider quirks in the driver, not in `.nl` source. If `max_tokens` is null, omit a provider field that requires an integer.

### Decision capability of any driver

A driver may carry `decide: DecisionScorer` (`Object.assign(driver, { decide })`, contracts.ts `DecisionRequest`): score a finite list of replies (`options`, each the JSON text of an allowed value) after `messages` and return `{ log_probs, tokens? }`, optionally with `scoreMany(items)` to batch. The runtime uses it for `readout: decision` functions, `decisionReadout: 'finite-returns'`, refinement judges and `decide(fn, ...)`; a driver without it, or whose scorer throws an error starting `decision-unsupported`, gets the ordinary tool loop for that call (the trace records `decision_readout` `phase: 'unavailable'`). Any other scorer error fails the call. A gateway driver implements it by asking the service that owns the model; endpoint drivers use `prompt_logprobs` or the llama.cpp fork's score endpoint. Wrappers copy it with the driver (`wrapDriver`); `DecisionScores.prompt_tokens`, when the scorer reports it, is the usage the evaluation `UsageGateway` charges. The readout's prompt is the call's opening with a short system turn (`DECISION_SYSTEM_PROMPT` plus the program's guidance), not the tool manual: the readout offers no tools. Wrap a scorer in `coalescingScorer` (`@natlang/node/model`) so readouts of concurrent calls reach `scoreMany` together.

### Pi as an injected dependency

`@earendil-works/pi-ai` is a peer dependency of `@natlang/node`; only `@natlang/node/model/pi` imports it. `createPiModelBackend(provider, model, environment, apiKeyEnv, headers, piOptions, modelOptions, mode, piPayload, models?)` takes a Pi `Models` registry as its last argument: a host that holds its own authenticated models passes them and natlang reads no credential file. Left out, the CLI default is used (Pi's catalog over `natlang auth` credentials). The request-to-Pi and Pi-to-turn translation is pure and imports no Pi: `toPiContext(request, model)`, `piToolChoiceOptions(api, request)`, `piDeltaTranslator()` (one per turn: Pi stream events to `ModelTurnDelta`), `piAssistantContent` and `piAssistantToTurn(reply, provider)` (`@natlang/node/model`, `model/pi-translation.ts`). They are typed over structural Pi shapes, so upstream pi-ai 1.4.0 and Vibestudio's `@panticonic/pi-ai` fork both satisfy them; a gateway calls them with its own Pi instance. The one cast at the boundary is `toPiContext(...) as Context` (tool `parameters` are JSON schema where Pi types TypeBox).

### The managed llama.cpp package

`@natlang/node/llama` (`src/llama/`, one self-contained bundle entry) is the managed llama.cpp machinery with host dependencies injected as a `LlamaHost`: `environment` (the `NATLANG_*` settings and `PATH`), `directories` (runtime, models, modelHashes, warmPrefixes, state, checkoutModels, templates), `processes` (`spawn`, `run`, `freePort`), `fetch` and `log`. `nodeLlamaHost(environment, log)` is the CLI's binding. It holds runtime discovery/install/fork build, the stock model list with download and SHA-256 verification, slot sizing (`planServerSlots`, `localSlotPlan`), the warm prompt-prefix cache (`warmOpening`, `openingOf`, `pruneWarmStates`), the shared-server record, and `createLlamaServer(host, options)`: `start()` returns the endpoint, `admit(request)` gates a turn (starts, restores or prefills the request's warm opening, counts it in flight), `plan()`, `status()`, `close()`. Who runs the process is the host's choice: the package spawns through `host.processes` (default `spawnLauncher`); a host with its own supervisor passes `launch(launch, signal)` and receives the exact command, arguments and endpoint; a host that already runs a server passes `attach` (`endpoint`, `slots`, `contextPerSlot`, `modelSha256`, `runtime`, `warm`) and nothing is started. `options.headers` go with the package's own requests (health, slots, warm states) for a server behind an API key. Decision scoring against that endpoint is the endpoint driver's: `openAICompatibleModelTurn({ endpoint, model, apiKey, scoreEndpoint: true, scoreAll: true }).decide` (`@natlang/node/model`). The package imports nothing from `model/`; `createResolvedModelSession` (model/local-server.ts) is the CLI composition over it.

## Folders

A directory reducer's first parameter is a `Folder` (or a `FolderHandle` from `folder.dir(path)`). Node's `openFolder(dir)` fronts a directory lazily; `saveFolder(dir, folder)` writes its committed changes back, each file atomically. A direct reducer call returns its typed value and discards file changes; `folder.apply(reducer, ...args)` keeps the committed ones.

A reducer call is a transaction: open the folder with `'overlay'` access (writes stay in memory), `apply` the reducer, and commit `await folder.diff()` yourself. `saveFolder` replaces files one by one, so for all-or-nothing durability write a log record with every changed file's new content first, then the files, and finish an interrupted commit from the last record on open (`applications/nldb`). Serialize writers; a read-only question is a direct call on an overlay, so anything it writes is discarded.

## Authority

Services, live values, and packages available to eval run with the application's authority. `workspace` selects the `package.json` whose dependencies eval and callable-folder code may import; `network` allows `fetch` in eval. A timeout or failed validation cannot undo an effect that already happened.

To initialize a soft body from text (a soft function, a soft skill, a system-prompt piece, or a standard-library body), use `initBodyInContext({ endpoint, store, text, type, render })` from the runtime package. `render(placeholder, driver)` captures the actual model request with the native-state placeholder; the native server initializes the body in that context and returns an `nzs1` artifact with source-token provenance and exact positional import receipts. The supplied archive retains the native state, and `.nz` files carry native `nzs1` and `nnt1` artifacts. The server must advertise the native text-write and contextual init-body capabilities. A missing producer capability or a render that omits the exact placeholder fails the operation; there is no text-emulation or projected-block fallback.
