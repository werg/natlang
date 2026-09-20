# Python and TypeScript runtime parity

Both interpreters implement the same natlang state transitions, typed actions, tool surface, source linking, and trace contract. The TypeScript host also has browser and Node adapters. Platform-specific I/O belongs in those adapters; it does not change portable interpreter semantics.

## Change rule

A change to `natlang` runtime semantics must include a TypeScript counterpart in `ts-host/src/native` or `ts-host/src/contracts.ts`, plus a paired fixture in `ts-host/test/native-parity.test.mjs` or `conformance/`. A TypeScript-only correction can use a paired fixture to demonstrate that Python already has the desired behavior. The same rule applies to prompt and tool schema changes. The checked source workspace and run-budget defaults are part of this contract.

The [parity workflow](../.github/workflows/runtime-parity.yml) runs the [change gate](../scripts/check_runtime_parity.py), the complete Python tests, and the TypeScript tests with `NATLANG_PYTHON` set. The latter makes differential tests mandatory instead of silently skipped. `npm test` also runs the shared conformance programs. The change gate is a review trigger; the differential and conformance tests supply behavioral evidence.

## Scope and known platform differences

- Python executes isolated QuickJS; TypeScript executes trusted code with a shared application object. Eval authority and host effects cannot be byte-identical. Paired tests compare portable state, actions, diagnostics, and traces with executor-specific fields normalized where needed.
- Node disk/process bindings and browser DOM/network capabilities are platform adapters. Browser model inference is a model driver over the same native tool agent.
- The legacy text-action grammar, `ModelAgent`, and Python bridge were removed. Structured tools are the supported interpreter surface in both runtimes.
- Explicit safety budgets remain available. By default inference turns/tokens/time and run episodes/depth/actions/tool calls are unrestricted; local creation and pending-node shape constraints remain language rules. Check [SPEC.md](../spec/SPEC.md) before changing one of those constraints.

When adding a tool or reduction behavior, update Python and TypeScript, add a paired fixture that reaches the behavior, and run `python -m pytest -q` plus `NATLANG_PYTHON=python npm test` from `ts-host` before merging.
