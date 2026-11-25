# Native cross-task capture and lexical entry grants

The Source35 `program-run-dedupe-audio-recording` capture has 21 records, one root and no missing or extra declared child IDs. Its authored `check` executes in a fresh program task while the recorded caller remains live. This is supported causal ancestry, not an ownership transfer.

Original manifest SHA256: `390f469a743ffe02144e6b873b8b9737df1ad6f620bf02e6f5e32bd06f576a6d`. Evidence is retained at `/home/werg/vibe-cross-task-custody-ops/actual-graph-inventory.json`; originals were not modified. The actual edge is `task-2-2f2ae0c2-8591-484c-b650-09b7af7bdd03/1` → `task-3-ac72fd17-8130-4015-942a-02f1ed58b371/1`. Authoritative owner kind/key and eval run match; the child starts and ends within the parent lifetime. Its native occurrence names the parent's exact occurrence SHA256 `37e70a3c51fe8d5104577ac68520bfebc901851f6ba3193ed192b69c32993e8b`. Both own-task custody hashes and their exact child sets remain intact.

## Inventory before the producer change

- Base `packages/natlang/src/programs.ts:295` binds loaded program functions through `runTask`, preserving the program's own services, declarations, scopes and root. SST `open-goal-program-run-managed.ts:175` loads and invokes the actual authored `program.check`. A new task is intentional.
- Core `runtime/runtime.ts` authenticates trace parents by live invocation and identical recorder ownership. `NatlangRuntime.run` creates a fresh task, preserving its lexical authority, model and cancellation policy. `retainClosureTraceParent` separately supports escaped callables; it keys associations by the actual parent ID rather than mutating a task-global parent slot.
- `registerGraphCall` attaches the child's ID to the authenticated external parent's child inventory. Sibling ordinals delegate to that parent. `crisp-trace.ts` records the real parent occurrence; `native-entry.ts` keeps the child's own task ID. `CallCapture.finish` hashes the actual task/call/parent/children/effect tuple.
- Agent, crisp-tier, authored TS entry, inline/named callable, bound callback, module/folder and iterator paths share those Core context/capture helpers. Node and Workerd use the same implementation. Existing nested-task, escaped-callable and crisp-occurrence tests cover truthful cross-task ancestry, explicit independent roots, foreign recorder isolation, closed/completed/aborted owners and lexical service separation.
- Base RecordsStore/NativeRecordsDO preserve the original per-call record and eval ownership. The actual archived producer evidence satisfies these contracts. It needs no invented parent, merged task ID or changed custody.
- Python/Node observed readers had incorrectly required every causal child to share its parent's task ID. The independent reader strand removed that assumption while retaining exact membership and ownership checks; the unchanged real capture now yields 21 calls and 26 eligible decisions. Direct replay remains a separate boundary: task-local `graph_order` namespaces must not be treated as a single portable effect order.

## Separate defect fixed

The inventory found that authored native entry capture used `recordingServices(task.services, ...)`, replacing a narrower lexical `frame.services` subset. Kernel agent/crisp paths already honored that subset. Native entry now uses `frame.services ?? task.services`, preserving explicit empty grants and recording wrappers. This did not cause the graph reader rejection and does not gate collection resumption.

Six new public compiled-module controls exercise narrow, empty and default service grants on Node and Workerd. They verify actual denied-port nonexecution, exact returned native values, crisp effect capture and zero model requests. Together with existing nested-task lifetime/causality, escaped-callable and crisp occurrence controls, all 30 tests pass. The official Node build and quick push gate pass under ledger unit `vibe-native-entry-grants-20261125a` (1.6 GiB peak).

No model inference, paid probe, historical record rewrite or child-inventory waiver occurred. The existing genuine four-call higher-order training proof remains unchanged. Full browser UI execution, new five-target SDK staging and portable cross-task replay were not rerun by this producer investigation.
