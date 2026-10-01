# natlang TypeScript runtime and compiler

This package builds `@natlang/node`, `@natlang/browser`, and the `natlang` CLI.
It contains the compiler (inline `nl` planning, callable-folder checks,
lowering), the runtime (tasks, the invocation kernel, callables, `iterateOn`),
the interpreter (the model's eval session), and the application utilities
(`EventLoop`, terminal shell, DOM renderer, playground projects). See
[native packages](../NATIVE_PACKAGES.md) and [development setup](../DEV_SETUP.md).
Instruction adaptation APIs, compatibility rules, and recorded validation are
described in [ADAPTATION.md](../docs/ADAPTATION.md).

## Build and test

```sh
cd ts-host
npm ci
npm run build              # Node build, then the browser bundle
npm test                   # build, applications, type checks, tests, conformance
npm run test:browser       # real Chromium smoke (needs a Playwright browser)
```

Node 22.13 or newer is required. `docs/TS_NATIVE_IMPLEMENTATION_STATUS.md` maps
the source modules.

## Use

```ts
import { createNatlangRuntime, loadNatlang, openAICompatibleModelTurn } from '@natlang/node';
import { handle } from './app.js';                       // compiled with `natlang build`

const runtime = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint, model }), services: { wiki } });
const report = await runtime.run(() => handle(ticket));   // natlang calls inside find this task

const review = loadNatlang('review/review.nl');           // a named function, without a build
await runtime.run(() => review(observations, criterion));
```

- `natlang build` / `buildProject` compile a project: they check types and callable folders, plan `nl` calls, generate `foo.d.nl.ts`, and emit JavaScript bound to a runtime. `compileVirtualProject` does the same in memory (browser pages, workers, tests).
- `createNatlangRuntime({ systemPrompt: "App instructions…" })` appends application instructions to the shared runtime prompt. A callback is also supported. Additions retain the ordinary tools and call-specific language guidance.
- `runtime.run(fn, options)` creates a task; `runtime.bind(fn)` carries it into callbacks from uncompiled code.
- A failed call rejects with `NatlangCallError`; traces are delivered to the `trace` sink.
- `defineNatlang(source)` creates a natural-language function from `.nl` text at run time.

## Trust

Eval runs trusted code in the application's process; it is not a sandbox.
Services and live values are passed by reference, and their effects are not
rolled back. Keep operation identities and observations in the application
when retrying external effects.

Browser specifics: [`BROWSER_CLIENT.md`](BROWSER_CLIENT.md) and
[`FRONTEND_APPLICATIONS.md`](FRONTEND_APPLICATIONS.md). Terminal applications:
[`TERMINAL_APPLICATIONS.md`](TERMINAL_APPLICATIONS.md).

### Native program improvement

`improveProgram` runs an editable natlang directory-reducer program over complete
source. Its root composes baseline measurement, one experiment and incumbent
selection. `folder.iterateOn` checkpoints immutable source and portable state
together; a decreasing work measure and an independent semantic progress judge
provide finite iteration without instruction-entry gas.

Packaged cases in `examples/program-improvement` exercise implementation, repair
and simplification with independently supplied expected results:

```sh
natlang improve examples/program-improvement/repair.json --out repair-run --model MODEL --server http://127.0.0.1:8081
natlang improve inspect repair-run
natlang improve export repair-run --out selected-source.json
```

`resume` and `step` retain recorded source/model/allocation identities. `adopt
RUN --into PROJECT` installs a completed independently checked result against its
recorded base; `recover RECORD` and `rollback RECORD` support interrupted
installation and restoration. The selected result contains all runnable source
files, validation evidence and a disposition; returning source does not install
it in the original folder.

Editing helpers cannot evaluate their proposals. The caller owns evaluation,
acceptance and selection; accepting a population parent's child does not make
it the incumbent. Locked test confirmation is available only at the top and
cannot be repeated with a revised candidate. Unknown transformation obligations
remain unverified, and a requested change requires changed source.
