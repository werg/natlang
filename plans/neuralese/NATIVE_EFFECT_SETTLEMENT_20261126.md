# Native effect settlement inventory

Observed: discarded RPC promises can outlive an NL return and its captured effect completion, on Node and Workerd adapter. No models used.

Shared producers: `native/effects.ts` wraps asynchronous Native service calls; `runtime/kernel.ts` creates invocation frames and finalizes captures; `runtime/native-entry.ts` wraps loaded TS entries, including recording opt-out; `native/runtime.ts` evaluates actions, including standalone use. `runtime/context.ts` propagates logical ownership on Node ALS and browser/Workerd restored frames. `runtime/runtime.ts` tracks child calls/task shutdown, which alone is too late for fixture verification inside a task.

Consumers: Base confinedRealm/evalNatlang and all five SDKs execute these shared Core paths. Outer Base Sandbox waitAll cannot establish an inner Root-return boundary. No engine-specific service-execution implementation is substituted. Native request/completion identities remain captured at invocation time.

Plan: logical effect owner separate from trace IDs, drain before asynchronous invocation/action completion, track for final task retirement as fallback. No synthetic TS spans. Synchronous TS values remain synchronous; a synchronous root that drops an asynchronous operation fails closed. Opt-out `settleNativeEffects:false` preserves legacy behavior. Rejections stay actual failed effects, never fabricated completions.

Checked/unexamined controls are recorded after execution. Historical physical/provider failures remain open; this finding does not attribute them.

The internal-import control found that virtual project local TS imports bypass the public entry wrapper. The shared compiler now retains authored helper promises: ordinary TS uses `own` without imposing constrained recursion policy; constrained helpers use their existing guard. Eval uses the same logical work hook. Merely constructing an unresolved Promise starts no execution and is excluded. Generators keep their iterator protocol; their complete stream lifetime is not newly claimed as joined. Async iterator consumers still own their actual awaited operations.

Actual controls before final syntax-preservation follow-up: 72/72 passed on Node and Workerd-platform Realm adapter, including discarded RPC, loaded TS nativeRoots false/no recorder, unrelated-await gap between two RPCs, standalone NativeRuntime and NodeNativeRuntime, actual abort/rejection, lexical callback authority/escaped function, cross-task/foreign/expired ownership, ONCE and refinement checks, existing root graph custody. Node/Core builds passed. Actual new Base confinedRealm/EvalDO SDK consumer and all five fresh SDK repros remain unchecked until staged by the release owner.

Synchronous boundary: explicitly asynchronous Native methods are refused before invocation for a sync TS root. An opaque unmarked method returning a promise can only be detected after invocation; that root throws and task retirement retains the real promise. No synchronous success or completed-effect record is fabricated. General generators and arbitrary external asynchronous resource lifetimes beyond executed Native methods remain outside this new guarantee.
