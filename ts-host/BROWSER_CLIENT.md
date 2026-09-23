# Running natlang in the browser

For event-driven interfaces, see [frontend applications](FRONTEND_APPLICATIONS.md).

The browser build (`@natlang/browser`, `dist/browser/natlang.js`) is the same
runtime and compiler as Node, with a local model loader. Natural-language calls
run in the page (or a worker you own); only inference runs in the model's
WebAssembly worker.

## Build and serve assets

Build `ts-host` with `npm ci && npm run build`. Serve these from
`ts-host/dist/browser/` (or the package's `dist/`):

| Asset | Purpose |
| --- | --- |
| `natlang.js` | Runtime, compiler, and model loader |
| `wllama.wasm` | Main inference backend |
| `wllama-compat.js` | Safari/Firefox compatibility worker |
| `wllama-compat.wasm` | Compatibility backend |

Serve WASM as `application/wasm`, use HTTPS in production or localhost in
development, and serve GGUF files with byte-range support. Cross-origin
isolation headers (COOP/COEP) enable multithreaded CPU inference; without them
the loader uses one thread. Publish the model's tool-call template beside it.

## Load a model and run natlang

```ts
import * as natlang from '@natlang/browser';

const { model, status } = await natlang.loadBrowserLocalModel(
  { kind: 'url', id: 'my-checkpoint', url: '/models/my-checkpoint.gguf', templateUrl: '/models/my-checkpoint.jinja' },
  { contextTokens: 8192, onProgress: ({ loaded, total }) => console.log(loaded, total) });
console.log(status.diagnostics.gpuSelectionReason, status.gpuFallbackReason);

const runtime = natlang.createNatlangRuntime({ model: model.turn, services: { appState } });
const project = natlang.compileVirtualProject({ files: { 'main.ts':
  "import { nl } from '@natlang/browser';\nexport async function main(): Promise<number> { return await nl<number>`Return seven.`(); }\n" } }, natlang);
const value = await runtime.run(() => project.require('main.ts').main());
```

- Sources: `{ kind: 'url' }`, `{ kind: 'files', files: [file] }` for a user-selected GGUF, or `{ kind: 'huggingface', repo, file, quant }`, each with an optional `templateUrl` or `chatTemplate`.
- By default the loader requests full GPU offload on a capable adapter and retries on CPU if loading fails (`cpuFallback: false` disables the retry; `gpuLayers: 0` selects CPU). Diagnostics report the requested layers; the backend does not report actual residency.
- `loadBrowserModelCatalog()` reads `/models/browser-catalog.json` (the published default and alternatives) for a model picker.
- `model.turnHistory` and `model.lastTurn` record per-turn duration, token counts, schema size, and malformed-call retries.
- Named functions load with `loadVirtualNatlang(files, 'main.nl')`; playground utilities (`runPlaygroundProject`, `projectEntry`, `traceFrame`) run and inspect small projects.

Compiled browser code restores the natlang task after every `await`, so calls in
ordinary async code find their task. Wrap callbacks invoked by uncompiled code
with `runtime.bind(fn)`.

Eval executes trusted application code in the page. Do not give untrusted
scripts privileged services, and do not assume that cancelling a call rolls back
a browser API call it already made.
