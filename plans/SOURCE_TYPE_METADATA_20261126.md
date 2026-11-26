# Reuse validated linked source types

The Source53 frontend cache correctly joined four first loads into one 86.8 second compilation. This does not explain the entire observed startup interval. The frontend already computes linked type-alias strings, but discards its per-load memo before the owner loads the validated sources and links those same type modules again.

| Part | Implementation and reason |
| --- | --- |
| Goal, semantic composition and callable visibility | Existing natural programs and their private folders; unchanged. |
| Exact source/root/compiler-option identity | Crisp hashing of the complete immutable project, including mounted sources/SKILL files; no policy decision. |
| Linked aliases | Reuse only already-computed string maps from the validated frontend product, with content integrity. |
| Owner construction | Fresh callable records, type-map objects, module instances, realms, services and context/SKILL loading; never cache these live objects. |
| Mutable sources and opt-out | A changed source/root/options/metadata invalidates reuse; raw source loading remains the reference path. |

Inventory includes Core compileProject/source-session/linked-types/virtual-project/default-platform consumers and the actual Base ProgramLoader and Native build RPC bridge. The bridge currently transports unknown JSON unchanged. Node, Workerd and browser interpreters share this mechanism; generation/scoring engines do not parse these compiler products. Exact exclusions and tests will accompany the completed change.

No new model calls, serving policy, retry, deadline or source-privacy boundary is introduced. The remaining startup time and failed-loader/Vitest teardown cause remain open; this change does not claim to explain either.

The owner adapter must validate the new context against `programProductInput`'s actual options, not merely the options echoed inside a product. `Base/packages/natlang/src/programs.ts:270` supplies the lexical program ID, target, emitted/validation-only mode, declared service types/scopes and declaration mode. File identity includes the complete linked workspace map, so mount identities and SKILL contents participate. Failed compilations publish no type context; the external-root regression remains TS6059 and cannot attest aliases.

Bridge inventory (Host67a1 source): `packages/service-schemas/src/build.ts:1083` transports the product as `z.unknown()` inside a strict outer envelope; `packages/service-schemas/src/evalNatlang.ts:21` accepts/returns unknown JSON. `packages/builtin/src/eval-engine/EvalDO.ts:4015` forwards the complete input with the active execution identity. `src/server/services/buildService.ts:123` forwards the entire product value. `src/server/nativeWorkspaceRuntime.ts:359` and `src/server/nativeCompilerProducts.ts:81` key/serialize the full request and return the full parsed product, with no nested property projection. A boundary regression is part of the paired Host adapter batch.

Engine exclusions: the linked-types producer/parser is shared by Node/core/Workerd/browser exports (`src/{index,core/index,browser/index}.ts`); managed llama, Python native and SGLang consume model generation requests rather than TypeScript compiler products, so no backend protocol changes apply. Native engine module realms and owner tasks remain fresh. The optimization has a source option and `NATLANG_SOURCE_TYPE_REUSE=0` reference mode.
