# Preserve a malformed native turn and let the executing program repair it

Policy belongs to the existing natural-language call: after the runtime reports
an invalid model reply, that same call takes its next step. The service does not
choose another answer, mask markers, synthesize a block, close a span or restart
the episode. Exact token/identity/state verification and HTTP/archive plumbing
are crisp helpers. Existing task oracles and protected splits remain unchanged.

1. Native decoding retains an internal exact trace for every negotiated recovery request (and existing explicit RL recording), including
   actual prompt inputs/import observations, sampled IDs/logps/query positions,
   forced/fed input roles, completed writes and actor/sampling identity. Valid
   ordinary replies keep their existing public response contract; malformed
   model-origin replies expose the full trace. Request/infrastructure failures
   still fail as requests rather than pretending to be sampled model turns.
2. A known marker protocol error finalizes a non-executable model reply with a
   typed failure descriptor. The descriptor binds its precise triggering draw
   (or actual existing length boundary), whether it was fed, and unfinished
   producing inputs. It never advertises an unfinished payload as stored state.
   Completed real state artifacts and all real input dependencies are archived.
3. Streaming and non-streaming carry the same typed failed turn, full sampling
   receipt and raw output. The client records it before the runtime emits the
   existing rejection-style feedback and asks the same call to continue. No
   already parsed tool calls from that malformed turn execute accidentally.
4. Extend the shared replay compiler with a versioned protocol-failure law:
   authenticate the exact marker/state-machine violation, all actual fed inputs
   and every sampled action. An unfed final draw is a likelihood target, as an
   existing unfed EOS already is; it is never claimed as an observed KV export.
   Unfinished producing input ranges retain their actual low-rank transform,
   while only genuine completed/archived writes create importable graph spans.
5. Preserve whole-episode inventory and occurrence checks across repaired turns,
   including completed producer/consumer edges preceding a malformed suffix.
   One shared optimizer update scores all sampled turns; no failure action is
   dropped or externalized. Ordinary zero-Neuralese episodes remain eligible.
6. Verify real both-port marker failures, typed HTTP/collector repair, actual
   cached-vs-graph action logps and downstream producer gradients, malformed
   receipt rejection and durable archive restore. Retain historical failures as
   diagnostic-only fixtures; missing historical sampled tokens are not invented.

This introduces model-facing rejection feedback. Before adoption it requires
48 live samples per variant or explicit debt in MODEL_FACING_CHANGES plus the
next teacher window; authoring/integration recovery guidance changes in the same
push. The current frozen full96 and program-header48 diagnostics stay immutable.

## Implemented contract and qualification

`native.protocol-recovery` explicitly negotiates the typed response. Existing
unnegotiated requests retain their former capture/storage costs and fail-closed
behavior. Sampling/4 and greedy structural-capture/2 are failure-only versions;
ordinary successful public replies keep their original shape. Original actor
bindings, normalized probabilities, imports and completed state artifacts remain
exact. The shared compiler authenticates the offending control draw separately
from its actual fed disposition. Normal whole-episode coverage stays mandatory.

The runtime records completed native writes before rejecting the failed turn,
so even unused writes retain their real graph occurrence. The same call receives
one rejection and continues; no transport retry discards a sampled turn. Native
view failure envelopes preserve every successful prefix chunk and the last
failed chunk/combine, with no final value ID. Their client archives all completed
artifacts before propagating the rejection and source turns.

Terminal ledger evidence (2026-10-16):

- `native-marker-complete-cpu-20261016`:68 PASS20.29s,1.1GiB peak;
  new both-port failure cases plus existing replay/occurrence regressions.
- `native-marker-cached-replay-cuda-20261016`:22 PASS44.46s,3.0GiB peak;
  ten malformed control variants per port, plus actual HTTP runtime repair,
  archive restoration, consumer→producer gradients and shared optimizer updates.
- `native-marker-production-admission-cpu-20261016`:2 PASS15.26s;
  actual HTTP failed turn→repair→consumer also admitted through production
  whole-episode RFT preparation, preserving every event/action/failure binding.
- `native-marker-view-actual-final-cpu-20261016`:2 PASS4.04s;
  real HTTP templated chunk failure, exact forced inputs and action-logp replay.
- `native-marker-compact-memory-cpu-20261016`:2 PASS3.14s;
  actual two-port codec plumbing, one compact and one raw capture per completed
  producer, no per-decoding-step tensor copies.
- `native-marker-final-node-20261016`:13 PASS; typed stream/nonstream handling,
  repair, literal-string isolation, integrity negatives and failed-view prefix
  archival. `native-marker-integrity-build-20261016` build passed;
  `native-marker-recovery-quick-20261016`22 ratchets+TS passed.

Strict cached-vs-flat logp oracles and replay-adjoint checks use FP32 on both actual backbone
architectures, explicitly promoting the tiny Mellum diagnostic and rebinding
its checkpoint identity. Separate original-BF16 Mellum service tests validate
actual marker failures and exact captured IDs/roles/logps/positions. These are
mechanism qualifications, not fullsize cloud, model quality or task-lift claims.
The consistent native normalization correction `cdfa40451` is included.

The ordinary 65-draw tiny CPU measurement retained12,562 bytes of scalar JSON
trace and zero additional tensor artifacts. Paired warmed medians varied around
49–52ms with/without capture; this is a nonisolated tiny timing, not a claimed
production speedup. Actual one-token compact producer measurements stored160B
compact+400B raw tensors for tiny LFM and288B compact+1,296B raw for tiny Mellum.
The additional raw capture is once per completed compact write (and a decoded
observation once per actual adapted import), through existing durable store
ownership. No full KV/cache is cloned merely to retain each sampled draw.

One expanded CPU unit with a1GiB claim hit its own cgroup limit; its2GiB rerun
passed at1.1GiB. Initial integration fixture errors (context size, greedy logp,
RNG argument and template prefix) were corrected without runtime gate/tolerance
changes. No production process was stopped for these checks. Historical frozen
failures lacking sampled-token receipts remain diagnostic-only. Independent
integrated review and the recorded live feedback measurement debt remain before
cutover; current frozen96/PH48 runtimes were not changed.
