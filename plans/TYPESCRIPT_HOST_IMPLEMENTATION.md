# Full TypeScript host implementation

## Audit of the infrastructure plan

The 17 implementation patches are present and pass the local conformance suite. The status document previously described the code delivered, but several plan gates were not yet evidenced: a paired live-model tools-v2/tools-v3 pilot, full application demonstrations for P01/P03/P04/P10/P13/P18, and a student pilot on the selected surface. The local one-slot Bonsai server is occupied by the existing teacher backfill. These are empirical gates; code and recorded-driver tests cannot establish them. The browser embedding is explicitly a portable subset. This host closes the desktop TypeScript embedding gap without claiming those measurements.

## Architecture

Use the Python natlang runtime as the semantic authority. A typed TypeScript package owns the application environment, model-turn callback, and optional crisp evaluator. A small JSONL bridge carries only portable source, typed inputs/results, model turns, traces and eval requests. A full interpreter port is not required to host all current natlang language features. The protocol is internal and versioned; the TypeScript API is the user-facing boundary.

1. `NatlangHost.run` creates one Python interpreter process per run. It accepts a program document or checked in-memory definitions, inputs, budgets/seed policy, an optional model-turn driver, engine selection, and an optional trace path.
2. The Python bridge delegates model turns to the TypeScript callback. It retains the normal `ToolAgent`, schema, type checker, call tree, combinators, effects, and limits.
3. `TypeScriptEnvironment` compiles TypeScript in both expression and authored-body modes. The default is fresh evaluation context. A retained shared mode binds the caller's native `host` object by identity, so successive crisp calls see the same objects. `self` and `args` are snapshots. Results pass the common portable and typed natlang boundary.
4. The shared evaluator is trusted host code, not a security sandbox. It has a synchronous timeout for CPU loops but no hard memory or async cancellation guarantee. Direct host mutations may survive a failed eval. The trace records eval and host observations; arbitrary host state is not replayed. The isolated QuickJS engine remains available for declared `fx` capabilities.
5. The TypeScript package includes a scoped desktop binding for files, bytes and argv jobs. Applications may supply their own native objects. It does not force every native value through the natlang tree.

## Acceptance

- Full runtime examples through the TypeScript API: nested functions, Map/Fold, checked definitions, model-driven actions, and crisp code.
- Real TypeScript syntax including generated constructs, with diagnostics and compile errors separated from runtime errors.
- Shared native object identity across eval calls and authored functions; fresh mode resets eval globals; disposed resources fail explicitly.
- Declared Python capabilities work through isolated QuickJS. Shared host authority is explicit in the run configuration and trace metadata.
- Node and Python failures, malformed protocol data, timeouts and client disposal produce bounded errors. No unsolicited retries of effects.
- Cross-boundary values reject undefined, cycles, big integers, binary and native objects. Pending values are only descriptive input snapshots.
- Build, type-check, package and end-to-end tests run without the live model server. The live paired model and student pilots remain separately tracked in the implementation status.

## Delivery

The Python bridge and `ts-host` package implement this architecture. The Node integration suite exercises the complete interpreter through the bridge, including generated TypeScript syntax, model tool turns, checked in-memory and file sources, trace capture, declared application capabilities, async Fold input, direct native object reuse in Map, and desktop file/process calls. The host keeps the existing Python semantics as its authority; it is not an independent TypeScript interpreter port. See [the package README](../ts-host/README.md) for installation, examples, and runtime limits.
