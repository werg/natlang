# Spike: Python over a natlang folder (2026-09-27)

A prototype for section D of [../../DIRECTORY_REDUCERS.md](../../DIRECTORY_REDUCERS.md), kept until phase 3 builds the
real thing in `ts-host` and replaces it. Delete this directory then.

- `folderfs.mjs`: an Emscripten filesystem that shows a folder to Pyodide through a minimal view, and `folderView`,
  a spike-only view over today's `Folder`.
- `natlang_policy.py`: the loop, recursion, exec and import policy, the `__natlang_finite` lowering, and the import
  hook for local modules.
- `natlang_py.py`: the `natlang` module's shapes (`nl[T]`, `wait`, `iterate_on`) over a host module.
- `watchdog.mjs`: the worker that interrupts a running computation.

Run (Node 24; `ts-host` built):

```sh
cd plans/spikes/python-folder
npm install --prefix . --no-save --no-package-lock pyodide@314.0.7
node --experimental-wasm-jspi functional.test.mjs   # file operations, pandas, sqlite, the folder's diff
node --experimental-wasm-jspi scale.test.mjs        # 10k files: laziness, coherence, nl in pandas, interrupt
node policy.test.mjs                                # what the policy refuses and allows
node --experimental-wasm-jspi module.test.mjs       # nl[T], wait, iterate_on, type text
```
