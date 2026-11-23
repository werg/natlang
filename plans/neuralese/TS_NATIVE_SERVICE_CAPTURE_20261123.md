# Native service imports use the invocation recorder

The actual Node CallStore reproduction returned a native result from a typed TS helper but recorded zero effects. `NativeRuntime` used recorded services for model eval while `natlang:services` resolved raw `frame.task.services`. This also bypassed the kernel's configured service-result refinements.

Inventory before change:

- `ts-host/src/runtime/kernel.ts:755` wraps checked task services with the current call's exact effect recorder and checkpoint acknowledgements, then passes services and a child frame to NativeRuntime at line816.
- `ts-host/src/native/runtime.ts:815` owns that invocation. Its supplied frame must be copied and bound to the same recorded services, without changing the caller's frame or wrapping services twice.
- `ts-host/src/runtime/modules.ts:229` and `:478` resolve ordinary and selected TS native imports through `lowered.service` / `selectedService`.
- `ts-host/src/runtime/lowered.ts:578` resolves the current frame on each access. Both import paths therefore receive the same fix.
- `ts-host/src/runtime/kernel.ts:1032` already binds recorded services for admitted crisp cases. That separate path remains intact.
- Node, Workerd and browser execute this shared host runtime. Model servers (managed llama-server, wllama, Python native and SGLang) receive model requests, not authored TS modules; see `serve/sglang_actor.py:1-22` and `serve/native_learning.py:1-5`. Their host-side executions receive the shared fix, without a serving-protocol change. `native/folder-python.ts` provides folder Python execution, not a second `natlang:services` importer.
- Learner/extraction/held/collector observe native call effects via this shared recorder; no new effect schema is introduced. All five Vibe runtime vendors must be updated before the native passthrough conversions are activated.

Regression coverage must use actual native execution and CallStore: hidden model service + explicit TS import, nested and concurrent calls, receiver identity, exact argument/result bytes, and async durability acknowledgements and required persistence errors before imported results return. No paid inference is needed. Natural-language policy stays natural-language; exact native I/O is crisp.

## Receiver parity and completed checks

The same service-wrapper inventory also found private getters executing with proxy receivers and frozen targets violating proxy invariants. `native/effects.ts` reads getters with their original receiver; `native/refinement.ts` uses a shell proxy, preserves original method receivers and unchecked method metadata/synchronous behavior, and retains configured async refinement checks. Actual capture/incremental/refinement tests passed 20/20. Existing `calls:false` disables call recording; empty service refinement declarations disable that checking layer. Required capture persistence remains enforced at the imported TS call site.

Stable vendor successor `e6f63390365a998f145d59e9d8fce5a7a770ca0b` contains the same fixes on Core3b. All five canonical Vibe vendors and the executable eval audit reproduce byte for byte: 279 pinned file hashes checked. Staged Workerd5/5 and eval5/5 actual execution tests passed, including private getters through both wrappers. Base vendor source64b18f649832bca350ab6f1eee28e0938d18f4af; System2736afa64ad75bb05ab8df5ee68b0dc60b780900.
