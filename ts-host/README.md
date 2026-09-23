# natlang TypeScript runtime and compiler

This package builds `@natlang/node`, `@natlang/browser`, and the `natlang` CLI.
It contains the compiler (inline `nl` planning, callable-folder checks,
lowering), the runtime (tasks, the invocation kernel, callables, `iterateOn`),
the interpreter (the model's eval session), and the application utilities
(`EventLoop`, terminal shell, DOM renderer, playground projects). See
[native packages](../NATIVE_PACKAGES.md) and [development setup](../DEV_SETUP.md).

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
