# `@natlang/browser`

The browser runtime and in-page compiler for natlang, with the local WebGPU/WASM
model loader (`loadBrowserLocalModel`), `EventLoop`, `BrowserDomRenderer`, and
playground project utilities. Serve `natlang.js` with its sibling `wllama.wasm`,
`wllama-compat.js`, and `wllama-compat.wasm` from the same origin.
## Adaptation

Import `parseAdaptation` and `bindAdaptation` from the browser root, supply embedded
program metadata and an explicit executor identity, and select a binding through
the runtime or task options. `adaptation: null` restores authored instructions.
The browser includes portable artifact consumption; optimization and evaluation
workers are Node APIs. See the [adaptation guide](../../docs/ADAPTATION.md).
