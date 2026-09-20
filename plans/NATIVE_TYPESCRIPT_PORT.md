# Native TypeScript interpreter port

## Correction of scope

The first `ts-host` implementation used a Python interpreter bridge. The native reducer now backs the default `NatlangHost` export. `PythonBridgeNatlangHost` remains an explicit adapter for isolated QuickJS execution and differential checks. The smaller `web/natlang_lite.mjs` remains separate.

## Required native parity

The target is a TypeScript runtime that runs without Python. It must read the same program/source forms and produce the same typed states, action outcomes, diagnostics, call identity, effects, trace events and quiescence behavior on the declared conformance corpus. Its eval executor can run in a fresh context or directly share a retained TypeScript application environment. Native application objects stay in that environment; only portable validated values cross into natlang state.

Implementation order:

1. Port type grammar, named scopes, fit/coercion, pending-node construction and state dump. Compare exact fixtures with Python.
2. Port checked function graphs, path/reference resolution and the tools-v3 surface. Check rejected writes and missing/Null/empty distinctions before executing agents.
3. Port the reducer for Lambda, Map, Fold and Iterate, including nested calls, budgets, quiescence/resume, stream waiting, and effect order. Implement the model-turn loop against a TypeScript callback and preserve the existing tool schema and result text where compatibility is required.
4. Port invocation seed derivation, traces/reconstruction and scenario replay. Use the same canonical seed vectors and semantic outcome contracts.
5. Bind the already-built TypeScript eval environment directly to the native reducer. Keep fresh/retained and shared-authority choices explicit. Port source/type/run meta-operations and bounded native Map concurrency.
6. Run the supported structured Python conformance corpus as paired differential fixtures. Keep the Python bridge explicit for QuickJS and comparison; make the native runtime the default export.

## Release gate

The release gate is the structured program references, `conformance/infrastructure_baseline.json`, paired Python tool calls and traces, and the native host suite. It includes malformed/rejected actions and suspended/resumed runs. The old text-action API and its fixtures were removed from both runtimes. A browser build must state which host APIs it lacks while preserving interpreter semantics for portable programs. The live-model and student pilot gates in `INFRASTRUCTURE_IMPLEMENTATION.md` remain separate empirical work.

## Current progress

The default `NatlangHost` (also exported as `NativeNatlangHost`) runs program, checked-definition, and `.nl` / `.ts` / YAML / JSON file sources without Python. It has typed values, nested references, source calls, the Lambda/Map/Fold/Iterate reducer, live Fold streams, declared effects, a model-turn driver, deterministic model seeds, versioned traces, and complete Map snapshots. It can use a fresh or retained TypeScript eval context and can share a caller-owned application object by identity.

`npm run test:conformance` drives 22 program references and the infrastructure baseline action fixture. Python `tests/test_structured_session.py` checks validation cases migrated from the retired text-action fixtures. The remaining codebase-triage program has a separate end-to-end native test exercising checked file loading, nested Map and Iterate, and crisp library calls. The package unit suite includes paired Python checks for finite crisp outcomes, typed writes, and rejected actions, plus native host, cancellation, effect, and snapshot tests. Authored native crisp functions can `await` application methods in the shared environment. `NativeSourceWorkspace` supplies versioned source description, type checking, and isolated asynchronous child invocation, including calls made through a shared host object.

These checks do **not** establish full differential parity. An optional native review stage can approve or withdraw a proposed batch before any tool executes. Its prompt and schema match Python across all three prompt variants. Proposal and review decisions and traces match Python in paired approval and withdrawal-retry tests, while full audit content and other failure variants remain unverified. `admitNativeTrace` checks captured outcomes, ordered actions, required calls, and effects without re-execution. Native crisp functions and `run_code` expressions can `await` declared asynchronous `fx` callbacks, but that syntax differs from synchronous QuickJS code. Nested workspace invocations share the parent episode budget. Independent pure Map slots can run with bounded parallelism when the caller opts in and the native environment is fresh with no shared host object. The native agent's default system prompt is checked byte-for-byte against Python. Complete tools-v3 schema fixtures for a leaf, checked codebase, and pending child match Python. Paired Python traces cover crisp and natural leaf execution, nested Map, Fold, Iterate, and a quiesced root resumed on a second run, including exact action order and state observations. The full surface across every path and state remains unverified. A browser bundle of the shared reducer and tool agent runs in-memory crisp and model programs and shares application objects; it has a Node smoke test but no actual-browser test. It lacks filesystem and process host APIs and a synchronous CPU timeout. The TypeScript default supports the structured tool API; callers that need isolated QuickJS use `PythonBridgeNatlangHost` explicitly.
