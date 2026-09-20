# Browser inference for natlang

## Decision

Use Wllama v3 for the local browser model adapter. It runs GGUF models in a browser worker, supports a model's Jinja chat template and native tool calls, can fetch from Hugging Face or use user-selected files, and offers WebGPU with a CPU fallback. The natlang agent already speaks OpenAI-style tool messages, so the adapter translates one turn without rewriting the runtime or embedding a model-specific chat template. The browser bundle ships Wllama's WASM asset alongside its JavaScript, plus locally hosted compatibility assets for Safari.

WebLLM has an attractive WebGPU and JSON-schema stack, but its published function-calling support is still described as preliminary. Transformers.js is well suited to general browser inference, but its broad pipeline API is a less direct match for GGUF tool-calling chat. Keep `modelTurn` injectable so a different browser engine can replace Wllama without changing the natlang reducer.

Sources: [Wllama v3 guide](https://github.com/ngxson/wllama/blob/master/guides/intro-v3.md), [Wllama API](https://github.ngxson.com/wllama/docs/classes/Wllama.html), [WebLLM README](https://github.com/mlc-ai/web-llm), [Transformers.js WebGPU guide](https://huggingface.co/docs/transformers.js/guides/webgpu).

## Runtime shape

- `BrowserLocalModel` owns or accepts a Wllama instance. It loads from Hugging Face, URL, or `Blob[]`, then adapts messages, public tool schemas, tool-call JSON, usage, seed, and cancellation into a natlang model turn.
- Loading requests all GPU layers by default (`n_gpu_layers: 99999`), Wllama's full offload setting. `gpuLayers: 0` forces CPU and a positive count caps offload for limited VRAM. `supportsWebGPU` reports browser capability, while successful full offload still depends on available VRAM. Safari compatibility files are served locally; Firefox compatibility GPU can be opted into despite its substantial performance cost.
- `BrowserNatlangHost` uses that model when a run does not supply `modelTurn`. It shares the native reducer and tool agent with Node, and propagates abort and timeout to local inference.
- `.nl` and `.ts` source parsing lives in one platform-neutral core. Node supplies disk operations; the browser supplies an in-memory virtual file map.
- Browser exports include the checked source workspace and lower-level runtime, as well as the high-level host. Application objects can be shared with crisp eval in the browser environment.

## Operational constraints

The first model load downloads weights and needs enough storage and RAM or VRAM for the selected GGUF and context size. Tool accuracy depends on the chosen model and its chat template. Multiple WASM CPU threads require COOP/COEP isolation; WebGPU itself does not. Dynamic eval requires a permissive CSP and executes trusted application code; browser capabilities may have external effects. The native library does not expose Node process APIs or a disk trace writer in the browser. The run result contains an in-memory trace that applications can save as a Blob.

The adapter is covered by a fake Wllama engine driving a complete natlang tool loop; the browser bundle, virtual file loader, and source workspace have execution tests. An actual Chromium interpreter smoke runs separately and in CI. The browser pilot page and optional live-model runner capture a natlang checkpoint's outcome, tool-call validity, prompt/cache/completion tokens, schema size, latency, and trace. The pilot points to the local v8 failures GGUF exported from the project's own LoRA run and loads its official tool-call template from `models/templates`. Use those measurements before treating tool-call quality or speed as established.

The first headless CPU trials loaded v8 but hit the 180-second task timeout before receiving a model turn, even after reducing the per-turn budget to 128 tokens. The pilot now enables cross-origin isolation and multiple WASM threads; remeasure that path and test real WebGPU offload on a browser with an adapter. The official LFM template requires argument objects in prior assistant tool calls, so the browser adapter converts OpenAI-style JSON argument strings before rendering later turns.
