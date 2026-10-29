# Four-sandbox recovery and utilization, 2026-10-29

This document records recovery milestones. The newest update comes first;
the later sections describe historical boundaries.

## Update, 08:41 Berlin: regression diagnosis and prompt robustness

Latest observed live steps were12930/11920/12336/10978, all advancing on the
591119-row source, with no observed cgroup OOM kills. The four learners remain
standalone; replicated distributed coordination is implemented but has not yet
passed full-layout qualification or been deployed. Do not describe these steps
as distributed outer-optimizer progress.

The complete matched graph-search development packet scored base6/8 versus
step8341 1/8. This is a real regression on that family, not evidence that all
capabilities or the latest checkpoints have regressed. The native and GGUF
paths agreed on254/256 sampled greedy positions, including every repeated-tail
position in that comparison. The pinned tokenizer template is byte-identical
to the official snapshot. Keeping earlier thinking is intentional and was
already enabled for the8341 evaluation.

A concrete prompt mismatch exists: ordinary training trajectories used a short
program header without tool schemas, whereas this evaluator used the full prompt
with schemas. On the same initial graph80 context, short prompts produced
complete calls while the full prompt repeated an incomplete search expression.
There was also a real transport bug: tool-less header serving returned complete
XML in assistant content, which the harness rejected. The isolated parser repair
executes those calls, but graph80 still returned null rather than the required
3. The prompt/transport findings do not fully explain the capability regression.
Four other short-prompt trials shared one server slot and hit wall budgets;
they are inconclusive, not four established task-quality failures.

Train on verified full and compact prompt variants while retaining the same
semantic tool contract and exact assistant targets. Recompute all graph import
and producer positions and keep supervision on the model's own outputs.
Protected related worlds stay protected across variants. The first40 variants
(19 ordinary actions and one genuine10-node/12-edge graph, each rendered twice)
passed static target/import checks and the complete protection union. Exact
content dedup and numerical graph qualification are still pending; the candidate
591159-row source is not admitted. Expansion beyond this pilot remains required.

Main b37c417a6 integrates one-time pending-group focus, preserving the complete
packed groups, required credits and saved optimizer/RNG/exposure. It is being
checked in a frozen worktree before rollout; the current live frozen runtime
does not yet implement it. The67 targeted verified idiom rows had no recorded
visits in the audited checkpoints, despite being present in the recipe.

A full-size diagnostic loaded actual8341 weights and optimizer histories and
used a real2762-token training row with369 positive targets. Production
in-backward updates and deferred updates agreed exactly on scalar loss, all153
Muon-routed gradients and every trained parameter, both in BF16 and under
NVFP4 with the checkpoint's saved activation calibration. Parameter-addressed
identical stochastic-rounding draws isolate the update schedules. This excludes
that specific update-path error on this row; it does not test every batch, FP8,
live-before-save identity, or whole-task behavior. Results are in
`/home/werg/data/dgxops/claude-recovery-20261028/data/training-bug-audit-v1/real-update-parity{,-nvfp4}.json`.
Continue investigating bugs rather than treating more data as an explanation.

## Update, 06:45 Berlin: four learners earning new graph exposure

All four learners have resumed on the591119-row source1aa7 and native runtime1f0
with normalized cohort fairness, preserving their own earned full states. Actual
saved checkpoints prove Big2 communication visits on every rank: peer1 step10500
has34 distinct graphs/457954 forward tokens; peer2 step9711 has410/5730905;
peer3 step9758 has2/25366; peer4 step8960 has36/484101. These checkpoints have
different ages and are not comparable throughput samples. The latest health
window observed all four advancing, approximately7.8k–12.9k physical forward
tokens/second each, without OOM kills; publication and batch shapes vary.

All four v7 publication watchers now use exact owner-epoch guards, CPU niceness15
and single numerical-library threads. Durable complete cloud-GET receipts, rather
than an owner-GET's transient progress phase, determine earned full verification.
This fixed duplicate readbacks. Owner verification no longer sleeps while holding
the shared retention lock. Full-state fallback continues when strict control
collection rejects a changing checkpoint pin. Current independently exported
new-source checkpoints include peer1 step10500, peer2 step9711 and peer3 step9758;
peer4 step8960 export remains in progress. Full workstation readbacks remain
separate obligations. The exact finalfair runtime/plan capsule and peer2's9419
retirement-rebind control capsule both passed private owner GET/hash.

Periodic retention remains disabled while scoped historical-epoch and paired
retirement integration finishes. Unrelated changing diagnostic JSON cannot offer
a globally atomic namespace: use explicit managed-consumer leases and fresh
per-file reference observations without claiming such atomicity. No deletion
through the new guard has been authorized from a diagnostic dry run alone.

Tranche001 verification is complete:112674 train and2269 protected test graphs,
with114943 exact producer-call and consumer-span checks. Full protection-union,
deduplication, custody, capacity and source admission remain to be executed before
exposure; conversion/verification are not training credit.

The local step8341 development evaluation exposes premature graph-search answers.
Its first four completed cases incorrectly returned null; the official base solved
those exact cases. One trace first exhausted8192 output tokens without a tool call,
then queried only the start node's neighbours and returned null. A focused cached
GGUF/native comparison agreed on the top prediction at32 sampled positions across
two exact contexts; this does not establish whole-task equivalence or a training
cause. Preserve this diagnostic and inspect actual generated tokens and source
exposure before changing the recipe or attributing the regression.

## Update, 06:15 Berlin: expanded recipe and exporter fencing

All four allocations remain assigned to useful training; rolling continuations
preserve each learner's actual full optimizer/RNG/exposure state. There is no
production distributed coordinator yet. Peer3 has resumed from its exact9756
parent on the591119-row source and the normalized all-lane fairness scheduler.
Its saved9758 checkpoint proves two newly added Big2 producer/consumer graphs
visited, with25366 forward tokens. Peer2 has drained at9419 for the same
scheduler continuation; peers1/4 still need the combined source/scheduler update.
These are operational observations, not comparable model capability scores.

The prior scheduler could postpone the first Big2 graph for2813 groups despite
its source being adopted. Required-token-mass service now balances ordinary,
urgent and communication cohorts using actual saved exposure while preserving
every required visit. The complete future schedule and limiting packed groups
are checked before each continuation; no optimizer or quota reset is needed.

The new per-role publication watcher binds the actual accepted owner, runtime,
source and selector. Its same remote scan checks boot ID, PID and start ticks,
and its step floor includes the actual resumed parent's step. Control archives
carry exact predecessor-retirement proof bytes and restoration maps. A failed
control derivation is recorded while the canonical full-checkpoint backup
continues. Independent full checkpoint GET, independent control GET and current
source/ops object presence remain separate publication obligations. Peer3's
reviewed v5 watcher is active; subsequent roles receive their own exact selectors.

Local recovery now also includes independently retrieved and full-state-qualified
peer1 step8341, SHA1b74798966c659b19c305bb84c7aa4d9c6576afae9f8b16d6ff7696b8c1f2c09.
The Big2 original archive has independently completed private owner GET/hash.
Tranche001 conversion has completed112674 train graphs and2269 protected test
graphs; verification is still running, so these are not admitted or exposed yet.

Remaining operational work is explicit: finish the four rolling updates,
confirm sustained throughput and fresh graph exposure, keep checkpoint exports
and owner readbacks moving, remove repeated retention scans and self-bookkeeping
references that can obstruct bounded retention, and qualify complete replicated
coordinator recovery before deploying distributed outer merges. The earlier
64MiB authenticated replication measurement does not establish full-model cost
or complete production recovery.

## Actual incident and retained progress

The cohost `edad6d6e-b02e-4cf8-a0d2-0a3129420a18` reset under the same ID.
Its boot changed and its current coordinator/worker directories disappeared.
The control-plane Running label did not reflect workspace or learner health.
Fresh snapshot attempts failed; the prepared direct cohort backup had not
completed an independently verified production cycle. The latest outer state
is unavailable, not recovered. The reset allocation was closed after checking
that the replacement boot contained no current useful state.

The three remaining workers were drained before their offline allowance ran
out. Immutable full checkpoint pins retain inner optimizer, RNG, exposure and
old pending/outer history:

| Worker | Final step | Full checkpoint SHA-256 |
| --- | ---: | --- |
| 1 | 4519 | `33b108bd9a4b5aa1232e133e3d3ba110f4d28088c784cd33e1a33d97e039d87c` |
| 2 | 3737 | `70efb401…` (full identity in the operational pin receipt) |
| 3 | 3886 | `b3969f39a79031bc7b121c2a7adec2cb40d814d5062b91041e74a26fd0777649` |

Worker 1 has the corrected 294747-row source. Workers 2/3 still need the exact
2545-row withdrawal applied or verified on continuation. Earlier correction
history remains part of the lineage. Historical held data is diagnostic;
fresh confirmation data remains protected.

Private uploads, cloud-side readbacks and independent workstation GET/hash
verification are different states. Consult the live export receipts rather
than assuming every pinned checkpoint has reached all three.

## Recovery being executed

Replacement cohost `b1662ef1-2514-49f4-be56-645e83a6712c` has a persistent
role owner, one 96GB-class RTX PRO6000 GPU and measured 16Gi host allocation.
The old reset cohost is closed, leaving four allocations. The replacement has
not yet earned training progress.

Worker 1's step4519 full-state standalone preflight passed with exact slot/pin
identity, unchanged corrected source/held inputs, both committed withdrawal
proofs and reviewed runtime inventory. Its persistent owner accepted the
restart. Actual new optimizer advancement and a fresh committed checkpoint
are the next evidence required; acceptance alone is not progress.

Continue productive standalone training on survivors while preparing a complete
new-outer reanchor. Preserve each survivor's own weights and inner history;
do not overwrite all peers with the highest-numbered checkpoint. The common
new baseline is an authenticated useful seed. Existing weight deviations are
initial conditions, not new optimizer updates. Unknown old queues stay in
the archived parent and must not be replayed or labeled merged.

## Robustness work and remaining qualification

- Persistent owners, boot/PID/start checks and per-worker optimizer progress
  detect actual failures. Use lock-free manifest/progress health observations
  rather than blocking the tensor merge lock for routine liveness checks.
- Truncated and corrupt fragment downloads now fail without installing bytes,
  clean temporary files and retry through the existing transport path.
- Bounded mapped initialization and streaming seed writes have run on the
  actual 16Gi host shape. Keep measured cgroup pressure visible.
- Complete new-outer trainer integration and real four-peer recovery tests
  remain required before distributed cutover. A primitive merge test alone
  does not qualify full-state continuation.
- Coordinator replication must retain global tensors, outer momentum/EMA,
  accepted pending updates, sequence/session and source-transition records.
  Verify crash recovery and measure its bandwidth, storage and ACK overhead.
- Full cohort export remains quota-independent recovery, with explicit
  per-filesystem space budgets and independently readable private catalogs.
  It still needs an actual complete production cycle and restore verification.

Operational receipts, ownership records, export intents and health observations
are under `/home/werg/data/lfm3-sft-20261028/production-v3-live/`.

## 01:54 Berlin update

All three survivors have resumed actual standalone SGD: observed steps
5000, 3999 and 4082; committed full saves were 4921, 3915 and 4000.
These are distinct peer histories, not a global distributed step count.
Each saved continuation is checked for corrected source/proof and its actual
historical parent. Per-rank private cloud publication is separate from the
serialized workstation GET lane. The newest verified workstation copy still
lags; a retained step4519 GET is in progress, and the previous parents remain.

Replacement b166 has verified step4519 and historical step1211 parents and
has received the expanded 480754-row source. Its actual schedule must differ
from the live peer1 continuation: cloning its RNG/cursor would not establish
useful additional training. Exact-parent scheduling comparison and target
QAT/memory checks precede launch.

Fresh-outer recovery is integrated in main `a2992cc7e`; 99 independent root
integration tests passed, including actual four-process HTTP/quorum tests,
a separate real native trainer transition-save test, common recipe identity,
publication closure and bounded parallel startup. This is preparation for a
real four-native-worker cutover, which has not happened yet. All workers must
use the same current source and common manifest; each retains its own newest
weights, optimizer, RNG and exposure. Coordinator replication is still being
implemented and is not qualified by these tests.

## 04:00 Berlin update: four productive standalone learners

The former peer1 sandbox `b1040a7d-7226-4c3a-afe5-f00afbaf71e6` also reset
under the same ID and lost its workspace. It was replaced and closed. Its
newest independently recoverable full checkpoint was step5500; at least 627
subsequent observed updates remain unrecovered. Failed snapshots do not cover
that loss.

| Peer | Current sandbox | Observed optimizer step | Active source |
| --- | --- | ---: | --- |
| 1 | `f3ad785f-9f0b-4716-905a-ed26f0a83e1e` | 6683 | 480754 rows, b9a |
| 2 | `ea17666b-084f-4236-b552-48efd43f9f74` | 6909 | 294747 rows, 105f |
| 3 | `3aba3399-4a8d-4e48-b8a9-069e653bf0ef` | 6995 | 294747 rows, 105f |
| 4 | `b1662ef1-2514-49f4-be56-645e83a6712c` | 5971 | 480754 rows, b9a |

These are four separate advancing histories. Distributed outer merges have
not resumed. All retain their own optimizer/RNG/exposure; the common new outer
session must incorporate useful existing deviations without fabricating lost
momentum or replaying uncertain old pending updates.

Peer1's expanded source initially received no appended ordinary-row exposure:
the pending schedule left those rows at the tail. Default ordinary-cohort
token-service fairness now addresses this. At exact step6680, all nine physical
packed checks passed on its H100 (three limiting groups across BF16, FP8 and
NVFP4 full mixes); the actual remaining schedule retained exact quotas and
matched the tested scalar limits. Its checkpoint-preserving continuation has
performed real SGD. Saved new-row exposure still needs confirmation. Other
peers have inert successor plans and staged source/cache dependencies; their
own newest checkpoint and target-device checks precede rolling adoption.
Peer4's earlier explicit rotation must retire through the recorded fairness
transition rather than overriding the new scheduler.

Recovery improvements already in operation:

- A private, independently GET/hash-verified 1.155GB warm recovery bundle
  includes current source/runtime dependencies and all 513 authenticated
  parser indexes. Base weights and full checkpoints remain explicit separate
  dependencies. The local warm restore took 5.38 seconds; this is not a network
  or learner-start measurement.
- Existing private checkpoint objects can now be fully retrieved and verified
  without contacting their disappeared sandbox. A real private canary passed
  after its original file was removed and with an invalid sandbox ID.
- Bounded per-rank export queues and dynamic fail-closed retention protect
  current cloud heads, independently verified fallback parents, in-flight
  publications and readbacks. Provider zero-byte tombstones plus fresh 404
  checks establish logical deletion, not physical backend garbage collection.
- Qualified local full-state recovery copies at steps2512 and4751 remain.
  Step4519 completed independent GET/hash verification but its requested new
  local copy was discarded at the filesystem reserve floor; do not call that
  copy retained. Peer1's new exact6680 pin has cloud PUT and full cloud GET/hash;
  its workstation readback is a separate pending obligation.

Coordinator replication is implemented in an isolated review branch and is
not deployed. It must cover accepted queues, state seals, source/session
transitions, clock rebasing, normal restart, bounded generation retention and
actual owned-role fencing. Remaining gates include concrete initial bootstrap,
replacement-replica placement after primary loss and measured overhead on the
16Gi host allocation. Healthy standalone SGD continues during this work.

The Big2 verifier's quadratic target-count loop has been replaced by an exact
linear count, and cold parsing now collects capacity geometry in the same
pass. Tranche000 has passed verification: 112455 training communication graphs
and 2739 protected test graphs. No candidate exceeds the current 108272
forward/KV capacity bound. Deduplication and exact raw-source recovery from its
retained archive remain admission work; these rows are not yet learner exposure.
