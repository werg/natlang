# Reusing natlang in browser applications

For natlang-driven event reducers and view plans, see
[Frontend applications](FRONTEND_APPLICATIONS.md).

`BrowserNatlangClient` is the application-level browser API. It owns one local GGUF model, loads its tool-call template, probes and selects WebGPU, optionally retries loading on CPU, runs native natlang programs, captures per-run model metrics, and closes resources. The lower-level `BrowserLocalModel` and `BrowserNatlangHost` remain available when an application needs custom inference or runtime wiring. Neither API uses the Python interpreter or a remote inference service.

## Build and serve assets

Build `ts-host` with `npm ci && npm run build`. A web app needs the browser entrypoint and its worker assets from `ts-host/dist/browser/`:

| Asset | Purpose |
| --- | --- |
| `natlang.js` | Browser runtime and model adapter |
| `wllama.wasm` | Main Wllama backend |
| `wllama-compat.js` | Safari/Firefox compatibility worker |
| `wllama-compat.wasm` | Compatibility backend |

When using the package export `@natlang/typescript-host/browser`, copy the three Wllama assets into your public directory and pass their deployed URLs to the client. This avoids assumptions about where a bundler emits JavaScript. Serve WASM with `application/wasm`. Use HTTPS in production or localhost during development. Cross-origin isolation headers (COOP/COEP) enable multi-threaded WASM CPU inference; the client chooses one CPU thread when those headers are absent. Serve GGUF downloads from the same origin when possible and support byte ranges for large files. Put the model's official tool-call template at a reachable URL or pass its text as `chatTemplate`.

```ts
import { BrowserNatlangClient } from '@natlang/typescript-host/browser';

const appState = { count: 4 };
const client = new BrowserNatlangClient({
  host: appState, mode: 'retained',
  wasmUrl: '/natlang-runtime/wllama.wasm',
  compatWorkerUrl: '/natlang-runtime/wllama-compat.js',
  compatWasmUrl: '/natlang-runtime/wllama-compat.wasm',
});

const loaded = await client.loadModel({
  kind: 'url', id: 'my-checkpoint', url: '/models/my-checkpoint.gguf',
  templateUrl: '/models/my-checkpoint.jinja',
}, {
  contextTokens: 8192,
  onProgress: ({ loaded, total }) => console.log(loaded, total),
});
console.log(loaded.diagnostics.gpuSelectionReason, loaded.gpuFallbackReason);

const controller = new AbortController();
const result = await client.run({
  source: { kind: 'files', root: 'main.nl', files: {
    'main.nl': '---\nreturns: Num\n---\nWrite the number 7 to return.\n',
  } },
  signal: controller.signal,
  options: { seed: { mode: 'compatibility' } },
});
console.log(result.outcome, result.value, result.trace, result.model?.turns);
await client.close();
```

The source can be a checked virtual file tree, a program value, or named definitions. Crisp TypeScript runs without loading a model. A `host` object is available by identity to crisp eval as `host`; its mutations are live application mutations. Pass an existing `TypeScriptEnvironment` when multiple clients or runs must share the same eval context; the caller owns that environment. `mode: 'retained'` retains eval variables within a run. Runs are serialized per client. A client will reject model replacement or close while a run is active.

`loadModel` also accepts `{ kind: 'files', files: [file] }` for user-selected local GGUFs and `{ kind: 'huggingface', repo, file, quant }`. `chatTemplate` or `templateUrl` can be supplied with any source. By default the client requests all layers on a capable WebGPU adapter and retries on CPU if model loading fails. Set `cpuFallback: false` to require the automatic GPU attempt to succeed; set `gpuLayers: 0` to choose CPU or a positive count to request partial GPU offload. The returned diagnostics report requested layers and probe details; Wllama currently does not expose an authoritative actual layer count. Use a fresh client or `unloadModel()` to release weights. Pass `signal` to a run or model load for cancellation; cancellation cannot undo external effects already performed by a crisp function.

`client.run()` adds `{ model: { id, diagnostics, loadMs, gpuFallbackReason, turns } }` to the native run result. Each turn records duration, token counts when supplied by the backend, tool schema size, and malformed-call retries. The result trace is portable JSON. No chat-template-specific messages are stored in source programs. For reviewed browser test cases and local training jobs, see the [playground](playground/README.md).

The eval environment executes trusted application code. Do not give untrusted scripts access to privileged `host` objects or assume that stopping a run rolls back a browser API call.
