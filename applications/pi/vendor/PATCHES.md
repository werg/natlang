# Patches to the vendored pi-durable

`vendor/durable` is pi-durable from https://github.com/earendil-works/pi at f10993b (`packages/durable`: src, test,
docs; MIT, see LICENSE). It builds against the npm releases of `@earendil-works/chord` and `@earendil-works/pi-ai`
(applications/pi/package.json). Its own suite runs with `npx vitest --run` in this directory; three test files import
other packages of pi's monorepo and do not load here (provider-session-cache-e2e, session-states,
system-order-cache-e2e).

Changes, each kept minimal:

1. Exports for the port's host (no behavior change): `appendAssistant` and `streamResponse` (harness/generation.ts);
   `fromSlot`, `toolDiagnostic`, `truncated`, `boundContent`, `publishProgress` and the `Reported` type
   (harness/tool.ts).
2. Pluggable scheduler policy and admission (PORT.md "The scheduler", owner decision 4). With neither option given,
   behavior is pi-durable's own: the inline policy and `admitSubmission` run exactly as before (the whole suite passes
   unchanged).
   - harness/types.ts: `HarnessOptions.schedulerPolicy?: SchedulerPolicy` and `HarnessOptions.admission?:
     AdmissionPolicy`; the facts and decision types of every policy point (`PassFacts`/`PassDecision` for R4, R4a and
     R9; `StepFacts`/`StepDecision` for R5; `ReconcileFacts`/`ReconcileDecision` for R8; `AbortTaskFacts`,
     `AbortConversationFacts` and decisions for R7), `OwnerStep` and `TaskFact` (a live task with its walk up the
     ownership tree), `SchedulerPolicy.applyCleanup?` (a decision's own cleanup for faulted and orphaned outcomes,
     instead of `settleSchedulerOutcome`), and `AdmissionRequest`/`AdmissionSession`.
   - harness/scheduler.ts: `TaskSchedulerOptions.policy`. With a policy, each policy point reads its facts on the
     Session line (`readOnLine`), asks the policy outside any commit, and commits the decision guarded on the records
     it read, deciding again when a guard fails: the pass guards each reserved or orphaned record (state, abort mark,
     version unchanged, no invocation, not completing, and for abort-mode and orphaned ones no live owned work); the
     step guards the task record and the closing flag; abortTask guards the record and its invocation;
     abortConversation guards the set of live tasks; reconcile's marks and withdrawals are idempotent, and each
     finalization is checked to be still held with no live owned work. Holding outcomes as `completing`, wait
     validation and the runtime commit gates stay crisp. Running a definition's `migrate` stays in the host: when it
     fails at reservation, the failure is recorded and the next pass decides (pi decides in the same pass). With a
     policy, idle waiters are resolved by the pass's idle answers (only when no task or conversation changed since its
     facts were read); `waitForIdle` adds the waiter and kicks a pass. A pass whose policy throws is retried after one
     second. `#terminate` takes the decision's cleanup.
   - harness/policy.ts (new): `crispSchedulerPolicy`, pi-durable's rules restated over the facts, with the ownership
     walks (`ownedLiveOf`, `inScopeOf`, `belowCancelledOf`). It drives the policy path in
     `vitest.policy.config.ts` (with `test/policy-setup.ts`, which selects it and a crisp admission for every Harness
     the suite opens): 963 of 967 tests pass. The 4 that fail queue an operation on the Session line between a policy
     point's read and its commit and assert where it lands (a runtime commit behind a handover, a handle operation
     behind an invocation's end, a context read behind a task's end, a second abortTask behind the abort handler); on
     the policy path such an operation lands before the decision's commit, whose guard then sees the change and decides
     again, so the invariants hold while the asserted ordering differs.
   - harness/harness.ts: passes the two options to the scheduler and to `Submissions`.
   - harness/submissions.ts: `Submissions` takes the admission policy; `submit` calls it with a narrow session
     (committed reads on the line, commits, Storage) instead of `admitSubmission` when it is given.
     `Conversation.reset()` and bound handles go through the same `submit`. pi's own `CompactionTask` still calls
     `admitSubmission` inside its commit; the port's compaction places background summaries through `durable.submit`.

3. Test support only: `test/chat-support.ts` `waitFor` reads its default timeout from `PI_WAIT_MS` (default 5000 as
   before). The conformance config (applications/pi/test/conformance) raises it, since the port's phases are model
   calls rather than microtasks.
