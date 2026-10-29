# Four-sandbox recovery and utilization, 2026-10-29

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
