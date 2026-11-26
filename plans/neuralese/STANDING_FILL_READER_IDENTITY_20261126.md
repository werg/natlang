# Standing fill reader ownership on controller resume

Actual aei1 resumed a compatible controller while its DGX/Colab fill workers
continued on an older reader. `reader-sources.json` records selected sources; it
cannot certify already-running child code. A present DGX collection returned
before rejoining its fill keeper. Fixed fill unit names also made reader changes
look like a still-present compatible worker.

The existing Cycle/AsyncRoundDriver fill ownership now uses source-specific,
fresh-attempt unit names and the existing memory ledger's admitted/queued unit
inventory. Exact fill-role matching excludes foreign roles. A resumed collection
joins one keeper for its actual actor; a reader change retires only stale fill
clients. The actor, policy, endpoints, learner, optimizer, RNG and preparation
stage are retained. Policy retirement, data/SFT yield and discrete-cycle cleanup
stop every actual owned source variant. No new controller, registry or handoff
channel is introduced. `NATLANG_EI_FILL_SOURCE_UNITS=0` preserves legacy naming as
a diagnostic opt-out; normal training uses the source-aware default.

Mechanism inventory: Cycle.fill_alongside and its recollection/data/actor cleanup;
AsyncRoundDriver.ensure_collection/_fill_beside; colab_fill_check; collection
migration/policy retirement/stop_collections; memory_ledger.owned_units and
unit_owned; filled_reasoning dead-process claims, exact-settings partials and
per-call served-policy attestation. Dedicated fill_window/pop_fill already have
explicit owned actor/session lifetime and are not standing-controller adoption.
Serving engines and TypeScript native/teacher runtimes do not implement this
Python offline data-consumer ownership mechanism.

Targeted helper/keeper and existing asynchronous-driver controls are pending.
Actual process/cmdline/source verification and safe live controller resume remain
required before claiming reader closure. The original two accepted episodes
remain unregistered until actual consumers are compatible. No actor/learner reset
is authorized by this change.
