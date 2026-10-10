# pi on natlang

A port of [pi](https://github.com/earendil-works/pi)'s durable coding agent harness (pi-durable) to natlang, unit by
unit (PORT.md has every decision; `port/` is the inventory). pi-durable stays the host for what is mechanism: the
Session line and SQLite storage, the scheduler's invocations and timers, the registry and the Harness API. Everything
that decides what happens in a run is a natural-language function, run by natlang's executor (a small, fast model):

- each phase of the three durable task kinds: `generation.nl` (prepare, request, classify, answer, the tool round,
  abort), `tool.nl` (lookup, repair, validation, hooks, intent, run, settlement, recovery) and `compaction.nl`
  (the cut, the summary, its placement);
- context derivation, the system prompt plan, the compaction cut and the token estimate, steering and follow-up
  selection at turn boundaries;
- admission (`admit.nl`) and the scheduler's policy (`scheduler/`: which task runs, step precedence, abort and its
  cascades, finalization, idle);
- pi's coding tools (`extensions/coding-tools`: read, write, edit, bash) and the subagent.

The agent is a provider model the harness calls through pi-ai (`ai.turn`). The functions are the harness around it;
none of them is the agent.

```
index.ts                 openPi(): pi-durable's Harness with the natural-language task kinds; runPiTask()
main.ts                  print-mode CLI and eval over tasks/ (the Node host)
browser.ts               openBrowserPi(): the browser host (see "In the browser")
types.ts                 every record the functions read and write, with its rules in doc comments
ops.ts                   durable.commit: the write operations, applied atomically with guards
host/                    services: durable, ai, tools, env, resources; the task kinds; policies; agent-models.ts
                         (the agent's provider); node.ts (Node envs and prompt), folder-env.ts and sqlite-wasm.ts
                         (the browser host's environment and session storage)
generation.nl generation/  prepare, request, classify, answer, startToolRound, finishToolRound, abort
tool.nl tool/            beginCall, run (+ exact result formats), fromSlot
compaction.nl compaction/  select, summarize (+ pi's verbatim prompts and transcript format)
harness/                 shared through uses:: context, planSystem, estimate (pluggable), deriveContext, cut, boundary
admit.nl scheduler/      admission; pass, step, reconcile, abortTask, abortConversation, cleanup
extensions/              coding-tools, pi-prompt (pi's seven sections, verbatim), subagent
vendor/durable           pi-durable at f10993b (PATCHES.md lists the changes)
test/                    scripted wiring tests; conformance/ runs pi-durable's own suites against the port
tasks/                   coding tasks for eval
```

## How a function commits

Natural language never runs inside a commit. A function reads committed state through `durable`, decides, and hands
the host one list of write operations with guards on what it read: `durable.commit(ops, expect)`. The host applies the
list atomically on the Session line with pi-durable's own helpers, so their invariants hold (every assistant entry's
usage is counted, a stored tool result is exactly what the model sees, a run's inputs settle with the run). A failed
guard writes nothing and rejects with "state changed"; the function reads again and decides again. The rules that
judge a commit's own staged writes (hold or terminate, wait validation, the runtime commit gates) stay in the host.

## Variants

Context building, the system-entry plan with the context estimate, the scheduler's policy and admission run on every
turn and every phase. Each has pi-durable's crisp code and the natural-language functions behind natlang's shared
`pluggable()` (`ts-host/src/runtime/pluggable.ts`), selected by a mode: `crisp`, `nl` (`natural-language` is accepted
as an older spelling) or `shadow`, which runs both, uses the natural-language result and records a `pluggable_shadow`
trace event saying whether the two agree.

- default: the four are crisp; everything else is natural language;
- `--context MODE`, `--scheduler MODE`, `--admission crisp|nl`: one at a time (admission has no shadow mode, since
  each side commits its admission);
- `--planning MODE`: the system-entry plan and the context estimate (prepare runs both every turn);
- `--pure`: `nl` for every point not set by its own flag, so every function in this app runs.

In shadow mode the scheduler's policy is compared with pi's rules restated over facts (`crispSchedulerPolicy`), not
with the scheduler's inline code; decisions agree when they match apart from their cleanup and the wording of their
messages.

## Running

```sh
natlang run --profile pi-executor applications/pi -- \
  --agent-endpoint http://127.0.0.1:8083 --agent-model nvidia/Qwen3.6-35B-A3B-NVFP4 "Fix the failing test"
natlang run --profile pi-executor applications/pi -- eval js-off-by-one --agent-endpoint … --agent-model …
```

The launcher's profile is the executor. The agent is any OpenAI-compatible server (`--agent-endpoint`,
`--agent-model`, `--agent-key-env`), registered with pi-ai as provider `agent`; without them the agent uses the
launcher's endpoint (the `--profile` it runs with), else the default profile's. `--session FILE` keeps the SQLite session; `--thinking LEVEL` sets the agent's thinking
level; `--quiet` hides the phase log.

`--agent-transport natlang` reaches the agent model through natlang's own model transport instead of pi-ai's
OpenAI provider (`host/natlang-provider.ts`): the request, content parts, tool-call decoding and token accounting are
those of natlang programs, so the agent speaks the Neuralese wire standard (spec/NEURALESE_PORT.md) exactly as they
do. `--agent-reader DIALECT` declares that the agent model reads that Neuralese dialect; it is checked at startup
against the server's `/v1/neuralese/info` (a dialect the server does not speak is a startup error), and the model
object carries it as `reader` (`{ kind: 'text' }` or `{ kind: 'neuralese', dialect }`), so code making a value for the
agent can choose its representation (plans/neuralese/DECISIONS.md, 2026-10-09). A Neuralese block in a message
(`{ type: 'neuralese', id }`, types.ts `NeuraleseContent`) reaches such a model unchanged; sent to any model that reads
text, the request fails with `neuralese-unsupported-backend`. There is no text fallback. Each request names the
conversation (its provider session ID) as the owner of its blocks (`x-natlang-owner`).

With `--companion`, a tool output longer than 2,000 characters is stored as a call of `view` (host/views.ts): the
result keeps its text form (the output, shaped past 6,000 characters by `--shaping`) with a reference to the call. A
text reader reads that text. With `--agent-reader DIALECT` the agent model instead reads the view's block in that
dialect, written by its server for the agent's intent at the call (and the note that `recall` returns the output),
once per call: memoized in the session, pinned on the server, restored from the runtime's Neuralese store when the
server lost it. The view is started right after the tool result; a view that cannot be written fails the turn
(retried by pi's retry policy when the server was unreachable or overloaded).

The companion also reads the agent's reply while it streams (COMPANION.md §7): what the reasoning names is looked up
before the turn ends (`--hints crisp|nl|shadow`: written paths and quoted names, or `hints.nl`), and a tool call is
prepared as soon as its arguments are complete. Only what the finished message keeps is committed; the next request's
companion section shows the notes. The phase log shows what was committed or discarded.

Help while the user types (plans/STREAMING.md §3): `openDrafts(harness, { env, helpers })` (index.ts) keeps a draft per
conversation as a document, changed by edit deltas (`drafts.edit(id, { from, to, insert })`); helpers (the
companion's `companionDraftHelper`, extensions/companion/draft.ts, policy `offers` crisp|nl|shadow) write offers
beside it (`drafts.read(id).offers`); `drafts.take` adds one to the draft and `drafts.send` submits it. Only the sent
message enters the transcript.

## In the browser

The same harness runs in a browser page or worker (`browser.ts`, `openBrowserPi`). The code is split in three:

- the platform-neutral core: everything the harness is (index.ts, host/, harness/, the task kinds, extensions/,
  vendor/durable without its Node files). It imports natlang as `natlang:runtime`, the neutral runtime specifier,
  which every build binds to the runtime it targets (ts-host `NEUTRAL_RUNTIME_SPECIFIER`, bound with the build's
  `runtimeModule` specifiers): the Node runtime under `natlang run`, the application builds and the tests (the Vite
  plugin resolves it too), the browser runtime in the browser build. No core module imports `node:*`;
- the Node host: main.ts, host/node.ts (pi-durable's NodeExecutionEnv per directory), host/resources.ts and
  extensions/pi-prompt/node.ts (context files and skills from the file system), SQLite through `node:sqlite`, the
  Neuralese block archive in a directory (ts-host FileNeuraleseStore);
- the browser host: browser.ts, over the core and the browser runtime (`@natlang/browser`).

In the browser:

- The workspace is a natlang `Folder`, mounted at `/workspace` (host/folder-env.ts, pi-durable's `ExecutionEnv` over
  it). read, write and edit change the folder; bash runs natlang's folder shell (just-bash, `runFolderBash`) over the
  same folder, with timeouts, aborts and the long-output spill (to the environment's own `/tmp`). Paths outside both
  mounts do not exist and cannot be written. A folder has no empty directories, symbolic links or file times: an
  empty directory made by `createDir` lives in the environment only, times are the environment's clock, and `watch`
  is not supported. Network access for commands is off unless `network: true`.
- The session is pi-durable's SQLite storage on sqlite-wasm (host/sqlite-wasm.ts), in the origin private file system
  through ts-host's `openOpfsSqlite` (pool `natlang-pi`, database `session` names it, default `pi-session`). OPFS
  databases need a dedicated worker; on a page's main thread, or where OPFS refuses, the session is kept in memory and
  `onReport` says why. A `Storage` or a sqlite-wasm database can be given instead.
- The executor is the page's natlang runtime. The agent model is reached through natlang's model transport (default
  here; `transport: 'pi-ai'` uses pi-ai's OpenAI-compatible provider): an HTTP endpoint, or the in-page WebAssembly
  Neuralese engine's endpoint (`startBrowserNeuralese`). A Neuralese reader (`reader: DIALECT`) is checked against the
  server as on the CLI, and its blocks are archived in the OPFS block store (ts-host `OpfsNeuraleseStore`).
- The companion (`companion: true`, or `{ shaping, hints, offers }` to choose its policies) reads, lists and searches
  the workspace through the conversation's environment, so it sees the folder; its search is the environment's grep.
- `pi.drafts` is the drafts API above: the page sends the draft's edit deltas while the user types, shows the offers,
  and sends the message with `pi.drafts.send`.
- The prompt has no project context files or skills (they are loaded from the file system on Node); its docs section
  names `packageDir` (default `/pi`).

Build (after ts-host's `npm run build:node` and `npm run build:browser`):

```sh
node ts-host/scripts/build-application-browser.mjs applications/pi    # applications/pi/dist/browser/browser.js
```

The build type-checks `tsconfig.browser.json` (the core and browser.ts, against the browser runtime's declarations)
and bundles it under ts-host's browser policy: a `node:*` module reachable from the entry fails the build with the
import chain. The runtime is left as the module `@natlang/browser`, which the page maps with an import map to ts-host's
`dist/browser/natlang.js`, so pi shares the page's runtime. `test/browser/index.html` is such a page (serve the
repository root): it runs a task against an executor and an agent endpoint on a small folder.

```js
import { createNatlangRuntime, openAICompatibleModelTurn } from '@natlang/browser';
import { Folder, openBrowserPi } from './dist/browser/browser.js';
const pi = await openBrowserPi({ natlang: createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint, model }) }),
  agent: { endpoint, model }, folder: new Folder({ 'math.js': '...' }) });
const { answer } = await pi.run('Add a function double(n) to math.js');
await pi.close();
```

## Verification

- `npm test` in this directory: scripted wiring tests. The crisp helpers are compared with pi's own code on
  thousands of random inputs (result bounding, truncation, transcript serialization, edit matching, image sniffing);
  the coding tools run through the runtime and are compared with pi's `CodingTools` (results, files, diagnostics);
  the policy path drives a real Harness through the scheduler and admission functions. The browser host:
  `browser-host.test.mjs` (the folder environment, the sqlite-wasm storage, pi's env service on the folder) and
  `browser-bundle.test.mjs` (the entry builds under the browser policy; the built bundle, loaded with the browser
  runtime and Node's `process` hidden, runs a coding task end to end on a virtual folder with a scripted executor and
  a fake agent server, the companion searching the folder; the session's memory fallback; the OPFS block archive with
  a stand-in OPFS). A Chromium smoke of test/browser/index.html is for browser CI. `stream-helpers.test.mjs`: the
  companion on the live stream (research started before the turn ends, a reset discarding it, a tool call prepared at
  completion, with a scripted natlang driver that holds its turn) and the drafts (deltas, offers, take, send).
- `npm run test:vendor`: pi-durable's own suite with the crisp defaults (967 tests; three files need other packages
  of pi's monorepo and do not load). `npm run test:policy`: the same suite through the guarded read-decide-commit
  policy path.
- `npm run test:conformance`: pi-durable's harness suites (generation, generation-recovery, compaction, context,
  prompt, inbox, submissions, tools, tools-recovery, structured, tasks) with the natural-language task kinds
  substituted, against a real executor (`PI_EXECUTOR_ENDPOINT`, `PI_EXECUTOR_MODEL`; `PI_CONTEXT`, `PI_SCHEDULER`,
  `PI_ADMISSION`, `PI_PLANNING` select a mode, as the flags do). The suites check exact entries and documents, so they test
  whether the functions are exact. Results: see Status.
- `eval` runs the coding tasks in `tasks/` on fresh git copies, judged by their check commands.

## Status

Measured on DGX (2026-10-08) with Qwen3.6-35B-A3B-NVFP4 on vLLM as both executor and agent, sharing the GPU with
a training job (about 170 generated tokens/s across all requests, about 28 per stream).

- Wiring tests: 25/25. pi-durable's own suite (crisp defaults): 967 tests.
- Conformance (pi-durable's harness suites against the natural-language task kinds, 322 tests). A long run keeps the
  code it loaded, so each run is stopped and restarted on the newest code when fixes land. Run 4: 118 of 128 pass.
  Run 5: 155 of 166 pass. Apart from the parallel-round timing deviation below, every failure of both either passes
  alone on the current code or led to a fix: a null head
  handed to pi's planner (a spurious system entry), a final provider or tool failure that was retried or cleaned
  instead of faulting, hooks and tool runs repeated within a phase (now once-effects), a registry missing pi's
  built-in tasks accepted, an eval built-in that collided with a function of the folder (`transcript`, every eval in
  compaction's summarize phase rejected), and functions that committed for their caller while answering `null` (now
  `{ committed }`). Remaining variance comes from the executor under load: a phase occasionally improvises and the
  second attempt, told why the first failed (`facts.previousAttempt`), usually recovers. Run 6 is in progress.
- Results of an aborted parallel round are written in call order: an aborted tool waits (bounded) for the slots
  before it, as pi's synchronous tools finish in call order.
- Live: `eval js-off-by-one` ran 7 generations and several tool rounds end to end (bash, read, edit), fixed the
  visible tests and missed the hidden `hasNext` case before its 90-minute limit. The subagent tool ran a child
  conversation end to end through `delegation` (the child ran bash, the parent answered with its findings).
- Speed: a phase is a multi-turn executor call, 1 to 5 minutes under this load. Planning (planSystem and the
  token estimate) has crisp twins behind the hot-path setting (`--planning`, default crisp), which cut prepare
  from 4–7 minutes to about 1.5.

- Known timing deviation: pi's "runs a round in parallel by default" test expects two 20 ms tool bodies to overlap.
  The port creates both tool tasks in one commit and runs them concurrently, but each body runs after its own
  natural-language call phase (minutes), so the two 20 ms windows overlap only by chance.
- Companion (`--companion`, COMPANION.md): first live comparison on `stale-build`, one run each side by side under
  the same load (2026-10-09). With the companion: done and passing in 94 minutes, 5 agent turns, 6 tool calls, 3
  briefings shown. Without: the build passed the check but the run had no answer when the 150-minute limit hit, after
  7 turns and 10 tool calls (it first read the wrong `build/` directory). One sample is an anecdote, not a result;
  repeated runs across the tasks are next.

General runtime changes that came out of the port: service arguments arrive as host-realm data; `ONCE_EFFECTS`
(the provider request and poll run once per phase); computed values keep fields their declared type does not
list, in returns and locals (the cause of executors retyping provider messages); union type errors name the
field; the default context budget follows the server's window; `natlangVitePlugin()`.
