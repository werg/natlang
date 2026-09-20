# Native TypeScript interpreter port

## Correction of scope

`ts-host` currently provides a TypeScript application host with a Python interpreter bridge. It is useful for shared native eval, but it is **not** a port of the natlang interpreter. The previous delivery must not be described as a complete TypeScript port. The existing `web/natlang_lite.mjs` covers a smaller portable fixture subset and likewise is not a full port.

## Required native parity

The target is a TypeScript runtime that runs without Python. It must read the same program/source forms and produce the same typed states, action outcomes, diagnostics, call identity, effects, trace events and quiescence behavior on the declared conformance corpus. Its eval executor can run in a fresh context or directly share a retained TypeScript application environment. Native application objects stay in that environment; only portable validated values cross into natlang state.

Implementation order:

1. Port type grammar, named scopes, fit/coercion, pending-node construction and state dump. Compare exact fixtures with Python.
2. Port checked function graphs, path/reference resolution, action parsing and the tools-v3 surface. Check rejected writes and missing/Null/empty distinctions before executing agents.
3. Port the reducer for Lambda, Map, Fold and Iterate, including nested calls, budgets, quiescence/resume, stream waiting, and effect order. Implement the model-turn loop against a TypeScript callback and preserve the existing tool schema and result text where compatibility is required.
4. Port invocation seed derivation, traces/reconstruction and scenario replay. Use the same canonical seed vectors and semantic outcome contracts.
5. Bind the already-built TypeScript eval environment directly to the native reducer. Keep fresh/retained and shared-authority choices explicit. Port source/type/run meta-operations and bounded native Map concurrency.
6. Run the Python conformance corpus as paired differential fixtures. Every declared supported feature must have matched final state, action outcomes, diagnostics, effects and trace semantics. Keep the Python bridge as a migration adapter until parity is demonstrated, then make the native runtime the default export.

## Release gate

Do not label the native port complete from a handful of examples. The baseline is `conformance/infrastructure_baseline.json`, existing canonical traces and the full Python surface tests. Include malformed/rejected actions and suspended/resumed runs. A browser build must explicitly state which host APIs it lacks, while preserving interpreter semantics for portable programs. The live-model and student pilot gates in `INFRASTRUCTURE_IMPLEMENTATION.md` remain separate empirical work.

## Current progress

The native structural type grammar and type fit relation now live in `ts-host/src/native/types.ts`, with direct tests for records, unions, named types, promises, and pending type constructors. The semantic interpreter, action surface, reducer, model loop and trace parity are **not yet ported**. The Python bridge remains the production path until the release gate above is met.
