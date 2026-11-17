# Source-authentic natlang module re-exports

A TypeScript barrel that re-exports an authored `.nl` default must load that same function's generated module. This is exact module plumbing, not a policy decision or a new natural-language stage.

Mechanism inventory before editing:

- `ts-host/src/compiler/project.ts:562` discovers source module edges and schedules `.nl` and `.nz` shims. It currently scans only imports; extend this shared discovery to static exports.
- `ts-host/src/compiler/lower.ts:613` rewrites generated shim and bound runtime specifiers. Extend the same transformation to exports, retaining clauses, type-only flags and attributes.
- `ts-host/src/compiler/node-project.ts:21` delegates Node builds to `compileProject`; `ts-host/src/runtime/virtual-project.ts:77` delegates virtual/browser builds to the same compiler. Both receive the fix. `ts-host/src/platform-defaults.ts:23` binds that shared virtual compiler to host platforms; `ts-host/src/core/index.ts` exports it to isolated/workerd hosts.
- `ts-host/src/vite-plugin.ts:35` loads a resolved `.nl` module ID regardless of whether its incoming edge is an import or re-export. It does not emit source-module edges and needs no parallel rewrite.
- `ts-host/src/compiler/project.ts:590` fusion binding discovery and `compiler/sequential-loops.ts` inspect imports for local bindings. Re-exports introduce no local binding, so these remain unchanged.
- Serving engines consume executable runtime calls after source compilation; this changes no request, capture or engine protocol. No engine-specific implementation exists for this source-edge lowering.

Scope includes `.nz` and runtime-specifier re-exports because they use the same edge discovery/rewrite mechanism. Preserve normal JS export semantics (in particular `export *` does not forward a default). No rewritten facade or replacement root is introduced.

Validation: execute Node ESM and virtual Node/browser barrel graphs against a scripted model, exercise `.nz` re-exports against real serialized assets, and verify namespace/type-only/runtime-bound export lowering. The authoring language reference documents the supported barrel form. The engineering findings register has no compiler module-edge incident requiring additional recovery steps.

Result: TypeScript build and all 40 tests in `project.test.mjs`, `compiler.test.mjs` and `neuralese-contexts.test.mjs` passed under the 2 GB ledger unit `natlang-module-reexports-tests-b-20261117` (876.4 MB peak). An initial browser test combined an await-restoring compiler target with the Node platform, which has no browser await hook; the regression now executes the barrel using a Promise chain on both targets. The source-edge mechanism, rather than host await scheduling, is what this test measures.
