# Engineering findings 2026-10-15 to 2026-11-02

Moved from AGENTS.md on 2026-11-09 (owner). Still binding for the code they describe; read the relevant section before changing it.

## Digest-cache maintenance stays bounded and off the source filesystem (engineering, 2026-11-02)

The shared append-only digest cache reached606.5MB and2.30million entries;
loading its full in-memory index exhausted a1GiB metadata job. Small operations
and tests use isolated hash caches. Periodic storage supervision compacts the
shared derived cache by default using a bounded SQLite index, dropping disposable
/tmp and /var/tmp entries and retaining the last observed signature per path and
key kind. Exact source signatures still control lookup; misses rehash bytes.
Do not stat-walk every historic source path on the external filesystem: that
attempt stalled on metadata I/O and was replaced. The successful live pass took
14seconds, retained337.7MB and used1.5GiB measured cgroup peak, so its storage
observer is admitted at2GiB. A recent-pass/growth check prevents repeated scans.
Concurrent cache additions may become misses; source data, checkpoints and
verification evidence are never modified by cache maintenance.

## Removed rows retain earned exposure on resume (engineering, 2026-11-02)

When resuming an existing SFT frame, restore its complete saved exposure ledger,
including rows no longer in the active source. Scheduling and quotas still use
only current entries. Filtering saved counts to current bindings silently erased
one historically trained, quality-deferred row in a zero-update save and correctly
failed the existing exact-admission guard. Preserve that guard, the intact parent
and correction evidence. The flawed save is diagnostic; do not continue it or
pretend historical exposure was undone. Fresh source phases retain their explicit
separate exposure semantics.

## Task selection observations survive controller handoffs (engineering, 2026-11-02)

Carry authenticated task-selection history by default across supervised/RL
controller handoffs. Bind an immutable source-controller snapshot and the actual
training packet; preserve each observation's original actor identity. Transfer
selector history only: no current-policy likelihoods/rollouts, trained rounds,
optimizer/exposure credit or behavioral qualification. Exclude absent task worlds
explicitly when projecting onto the current training packet. Newly sampled
attempts still come from the actual selected executing checkpoint.

## Policy episode order preserves task and role diversity (engineering, 2026-11-02)

Fresh native policy sources mix complete episodes by explicit task identity and
proportional student/teacher role mass. Do not group every attempt of one task
into consecutive updates and leave all privileged teachers as the final update.
Keep every graph, action, objective term and source custody unchanged. A resumed
policy source retains its exact earned prefix through arrange_and_seal; only
newly materialized work can be reordered. Record actual sealed order separately
from actual packed/update geometry, and report unknown coverage honestly.

The diagnostic five-update72 batch had20/20 correct attempts on four training
worlds, zero task-outcome spread, small bounded efficiency advantages and mostly
privileged-KL scalar loss. This is not evidence of task-outcome learning or an
established cause of the protected12/16-to11/16 change. Preserve its traces and
use actual broader task behavior when selecting the parent and next curriculum.

## Actor publication shares the live checkpoint reservation lock (engineering, 2026-11-02)

Post-commit actor-view pinning must reuse the exact live target lock owned by
an async checkpoint's spare reservation. Opening that lock path again conflicts
with the process's own flock. Validate the passed reservation target and lock
inode, retain its lifetime lock after pinning, and bind the captured async spare
rather than racing a mutable policy reference. A synchronous committed reservation
may already be closed; use ordinary locking then. Keep model-only derivative
readback distinct from decoding a full checkpoint for metadata publication.

## Rollback retains exact source-adoption execution authority (engineering, 2026-11-02)

A zero-SGD source-adopted checkpoint can still require its matching external
execution-transition controller and selection proof. A trained successor may
have folded that admission into its saved history; rolling back to the original
candidate does not inherit the successor's authority. Snapshot and bind the
actual original source-controller bytes and exact selected candidate. Preserve
the startup guard; do not clear pending markers or borrow a different parent's
admission to force a resume.

## BF16 SFT occurrence custody and semantic target quality (engineering, 2026-11-02)

Physical prefix aliasing changed BF16 SFT cotangent accumulation and actual
optimizer updates. Ordinary BF16 SFT now retains original gradient occurrences
while reusing independently qualified fixed-linear primals. Capacity uses the
same execution predicate. Old physically shared capacity evidence does not
qualify the larger original-row geometry; remeasure actual complete graphs.
The existing forward/gradient tolerances remain unchanged.

SFT68 (433e3ad4) and its stopped SFT69 successor (5fc60ab2) are diagnostics,
not production parents. SFT68 scored1/4 on protected development; BF95 still
passes case88 under the same runtime17 where SFT68 failed. Continue from the
better demonstrated BF95 state after actual source/runtime admission, without
claiming the sole cause of that regression is established.

The TRAIN-only semantic-ticket review found69 complete graphs supervising
keyword/regex substitutes for semantic judgment; eight other candidates were
false positives. Corrected401623-row sources preserve every other graph's
binding/order and held data. One removed graph had an earned visit; preserve it
as historical exposure. This targeted review does not certify the entire corpus.
Do not fabricate replacements or relabel historical actor provenance. Bind
Python reviewer, JS checker and actual TypeScript parser dependencies, and
apply authenticated ordinary/native whole-graph deferrals to the actual parent.

## Frozen runtime storage stays deduplicated (engineering, 2026-11-02)

Use the common immutable runtime file pool when no checked readers require
isolation. An isolated pool is temporary operational storage, not a reason to
retain another complete Node/dependency copy forever. During a verified idle
window, consolidate frozen trees with `seal_runtime_tree` into the common pool,
then prune only unreferenced old pool blobs with `prune_unreferenced`. Preserve
all snapshot paths, exact contents, protected inputs and historical receipts;
record the new storage observation separately. Do not repool shared inodes while
checked runtime readers are active. Reader scans are point-in-time observations,
not globally atomic coordination with arbitrary future launchers. The qualified
2026-11-02 consolidation retained47 frozen trees and retired15.48GB of redundant
old-pool file bytes. No source/training data or checkpoint was discarded.

## SFT update-budget references follow the selected SFT frame (engineering, 2026-11-02)

Do not inherit an explicit scored-graph budget receipt from a discarded diagnostic
branch when preparing a continuation from a different earned parent. The SFT
frame inside an RL checkpoint owns this reference, not the surrounding RL frame
or the last launch configuration. Verify the actual saved frame. When it has no
captured reference, retain its saved accumulation for the first real SFT update,
let the trainer capture the actual scored-graph anchor, then promptly adopt the
smaller accumulation with that earned anchor. Do not transplant a diagnostic
receipt or weaken the source/checkpoint guard to make a launch pass.

## Preserve admitted policy identity across ordinary resumes (engineering, 2026-11-02)

An unchanged CE objective resumes with its exact admitted policy binding,
including historical objective-transition declarations. The objective resolver
recognizes an already-adopted objective and records no new transition. Removing
an inert declaration changes the policy hash and can incorrectly re-admit a
quota basis measured against an older source/exposure frame. Do not weaken the
quota guard: preserve the admitted bytes, or measure and declare a real new
objective continuation against the actual earned frame.

Smaller SFT accumulation adopts the all-group optimizer update budget: scale
ordinary backbone and interface LR/latent LR by actual scored graph count over
the earned reference window, capped at one. Fixed loss normalization alone does
not bound normalized Muon/Adam parameter updates. Bind the reference to verified
row exposure deltas and the complete TRAIN scoring census. Preserve historical
reference/source custody across admitted appends. This bounds the LR budget;
it does not reproduce the larger batch's optimizer or scheduling trajectory.

## Reuse dense physical capacity, recheck current numerics (engineering, 2026-11-02)

Dense LFM3 continuations retain authenticated measured capacity-proof references
in the actual SFT frame before releasing restored metadata. A new checkpoint hash
alone is not a reason to repeat the complete largest-graph backward. Reuse only
when source/request semantics, parser/index witnesses, runtime, backend,
parameter and optimizer-storage shapes, execution layout and available memory
and scratch satisfy the measured contract. Run fresh small numerical/offload
parity at the current weights. Retain the original measured checkpoint and record
the current checkpoint separately; reuse is not a new measurement or task lift.
Historical proofs without complete custody and genuinely changed geometry still
require measurement. Relocated output/cache paths do not imply changed data, but
both actual request bindings and unchanged witness identities must verify.

Performance claims follow the actual enabled arithmetic. The BF16-rounded-weight
three-product adjoint improves a supported BF16 precision branch; it does not
establish a speedup for the current IEEE FFN recipe. Keep actual mode/dtype branch
witnesses and full-model gradient qualification. Exploratory kernel tolerances
must not replace the declared production certificate tolerances.

## Checkpoint input scratch obeys recovery storage admission (engineering, 2026-11-02)

File-backed activation-checkpoint inputs use the same per-filesystem admission
lock and configured free-space floor as full-state checkpoint writers. Physically
reserve each backing file before mmap/copy; a sparse truncate does not protect
space from concurrent writers. On the DGX continuation bind the40GiB NVMe floor
explicitly. Keep bounded pinned transfers, exact stride/value/gradient/RNG
semantics, graph-owned file lifetime and abandoned-store cleanup. The measured
whole-graph capacity gate must include the actual file-storage path and release
its scratch before relying on checkpoint-write headroom. Insufficient storage
fails admission without spending the recovery floor or deleting recovery state.

## Bound FFN saved activations with full execution custody (engineering, 2026-11-02)

Token-local FFN chunking alone bounds temporaries, not saved expanded activations
across chunks during an outer layer-group checkpoint recomputation. Long
chunked gradient-enabled FFNs now checkpoint each chunk by default at16384 rows.
Retain exact invocation functional weight substitution, native actor/reference
and cotangent scopes, gradient partitions and RNG. Merely restoring actor
metadata caused a real saved-tensor-count mismatch and was repaired before
integration. Force the branch in small-model certificates and record that
witness; a passed fixture which never exercised the path is insufficient.
Validate actual complete long graphs before claiming the capacity blocker fixed.

## Capacity must measure actual learner arithmetic (engineering, 2026-11-02)

Ordinary SFT's prepared capacity and numerical-reference probes previously used
VJP mode even when the bound learner selected plain cotangents. Both paths now
resolve the actual backend, including the native backend alias, as policy
capacity already did. Keep the backend and runtime code bound in capacity
identity; historical VJP evidence does not establish plain-mode capacity.
A failed qualification is not an optimizer update. Preserve the actual earned
checkpoint and test complete graphs under the corrected execution before
omitting data, increasing budgets or claiming the memory blocker is resolved.

## Actual replay qualification and quota-aware continuations (engineering, 2026-11-02)

Source-v1 native actor and packed training replay demonstrably disagree: the
complete earned67 diagnostic episode reached a 0.175524 action-logprob error.
Do not reuse those captures for production policy updates. The zero-SGD
`ffn-stable-v1` candidate preserves weights, optimizer/RNG and exposure while
making projection arithmetic and pointwise rounding consistent. Its real CUDA
forward/backward certificate and complete communicating-graph replay passed;
13,190 action probabilities differed by at most 9.54e-7. Qualification does not
establish capability lift: the paired four-case development result was parent
4/4 versus candidate 3/4. Preserve that disadvantage and its false-blocked trace
in selection evidence; do not call it nonregression or train on protected traces.

Inline precision training composes with the exact declared native FFN forward:
consume its effective quantized operands and retain invariant projection
arithmetic. Arbitrary instance forwards remain unsupported. Canonical projections
must not acquire a training-only vendor FP4 GEMM. Install precision wrappers
before packed capacity qualification; disabling precision is not the integration
fix. Execution handoffs retain immutable controller snapshots so later head
updates cannot invalidate the launch's historical source evidence. Within one
nested admission operation, a complete fresh byte hash can be reused only while
resolved path and device/inode/size/mtime/ctime remain unchanged; independent
operations start fresh.

Loss-mixture reports must account for remaining per-row visit quotas and actual
earned credits. One-pass token shares substantially understated Native loss mass
under seven-visit communication quotas. Use the prepared quota-aware 401,685-row
continuation and its exact ledger basis, preserving earned exposure and all
remaining visits. The 472 original-source render obligations were resolved as
quality exclusions with retained sessions; the seven losslessly graphified long
contexts still need capacity qualification and rebasing onto the actual adopted
quota-aware predecessor. Neither prepared package is earned exposure.

Portable development evidence must retain exact original evaluator files and
source/container path mapping. Seal on the source host and validate the copied
manifest; never rewrite raw score paths to manufacture verifiability. Structural
ZIP deltas are the default actor-transfer optimization when smaller than the
other verified choices. Preserve exact base/target hashes and independent
reconstruction readback; a compact delta is recovery only with its actual base.

## Native policy arithmetic and source preparation (engineering, 2026-11-02)

Actual native actors execute `shared_state.native_step`; CUDA BF16 projections
use the shared invariant kernel. A surrogate `F.linear` probe does not establish
native actor behavior. SGLang normalization, convolution, FFN and attention-gate
rounding defects were found and repaired, with a shared-projection candidate
prepared. Those fixes are not whole-model qualification: compare real states,
sampled probabilities and complete graph replay before admitting backend rollouts.
Do not relax numerical tolerances to make incompatible actors pass. New same-actor
policy updates check every selected-token log probability in a complete episode
before backward or SGD, retaining the actual checkpoint and diagnostic receipt.

Prepared acquired-source appends do not constitute adoption, exposure or capability
lift. The 34,152-row OpenResearcher delta is source-reported positive general web
research, not locally oracle-verified Natlang/coding demonstrations. Preserve that
identity and the explicit unresolved render/context obligations. Continue directly
from the actual corrected earned source; unadopted intermediate recipes earn no
history or stage credit. Reconcile launch parameters against actual stage settings:
an improved recipe alone does not correct inherited seed parameters.

## Exact inference-view transfers (engineering, 2026-11-01)

Use `scripts/coreweave/send_actor_transport.py` for new workstation/Pop actor
deliveries, with the bound transport SHA and the peer's persistent staging root.
It automatically uses the smaller verified XOR delta when a previous view is
available, otherwise the full view. The receiver reconstructs and rereads exact
bytes before publishing; original actor and full-state custody remain unchanged.
The peer cache is a performance hint, not a recovery or training qualification.
Keep actual bases while a transfer consumes them; missing cached bases use full
transfer, while corrupt bases are diagnosed rather than silently accepted.
Prepare the transport under its exact target serving Python runtime. Its source
inventory is authenticated independently of tensor equality: a transport prepared
on newer main cannot be loaded by an older pinned evaluator. Preserve failed
launch evidence and rebuild with the intended runtime; never weaken that check.
Preserve readback receipts and avoid rewriting already matching metadata while
actors use it. A full3.46GB view64→75 reconstruction verified a138.6MB delta
(4.01% of full size). This is measured transport saving, not model improvement.

## Honest stops, bounded efficiency and live escalation (owner, 2026-11-01)

Engineering update, 2026-11-02: terminal `blocked`/`failed` reports with a reason
hand off the live invocation when an explicitly configured fallback and budget
remain. Preserve scope, prior effects, actor provenance and graph-wide limits.
No fallback is allocated or paid provider enabled implicitly. Mixed-actor
handoff traces remain ineligible as same-policy whole-graph RL samples.

New scoring records at most 0.02 credit for a terminal failure report quoting a
linked, observed tool-error line from that invocation. This establishes grounded
reporting, not task impossibility; unsupported overwhelm/missing-input claims earn
nothing. Keep task quality zero. Recompute the credit at replay admission.
All-failed groups may learn centered reporting differences without standardizing
them; apply the existing bounded auxiliary-only optimizer scaling. Correct task
completion remains primary. Historical captures retain recorded reward policy.

Correct task completion remains the strongest outcome signal. A genuinely
blocked stop is correct only when the task oracle verifies the expected terminal
status, honest stop and its evidence; self-reported difficulty alone earns no
success credit. Preserve raw results and historical scores when repairing a
derived classification, and never silently discard earned optimizer updates.

New scores bind `bounded-rational-v2` efficiency policy: call, token and repeat
components use bounded count/(count+scale) costs, retaining resolution beyond
eight calls and2048 tokens. Every one of five actual recent trajectories had
saturated the old .03 call/token cost despite8--33 calls and6539--20259 generated
tokens. Replay honors the exact clipped-v1 formula for historical rows without
a policy field; never reinterpret their scores or earned updates. New-policy
rows bind the explicit identifier and unknown policies fail closed.

New whole-graph RL scoring includes bounded execution-efficiency cost by default:
at most 0.05 for observed calls, generated tokens and unchanged repeated actions.
Normalize task outcomes separately; do not amplify tiny cost differences through
standardization or teach all-failed groups to prefer cheap failure. Missing token
counts remain unknown. Historical bound rounds retain their recorded cost policy.
For newly prepared objectives, wholly efficiency-only accumulated updates also
scale the actual optimizer learning rates by their bounded advantage magnitude:
gradient scaling alone does not bound Adam/Muon steps. Restore ordinary rates
after each update and record adoption and the applied scale. Mixed outcome or
privileged-teacher updates retain their ordinary rate; their efficiency component
is not separately bounded in parameter space.

Inference escalation continues the actual live invocation through explicitly
configured model profiles. Preserve scope, completed effects, authority, graph
ancestry, remaining limits and actual per-turn actor identity. Never restart a
task merely to hand it off, reset budgets, or reuse another model's native state
or adapters as though compatible. Learning must reject unsupported mixed-actor
likelihood attribution. No paid generation is implied by configuring this machinery.

## Source continuations and asynchronous checkpoints (engineering, 2026-11-01)

Direct supervised continuations initialize and validate their resident recipe
plan before allocating the model. A configured resident directory alone is not
an initialized source-update authority. Preserve committed resident generations;
never replace a different recipe/runtime plan to get a launch through. Rolling
checkpoint custody snapshots are namespaced by the actual snapshotter identity
as well as the full checkpoint SHA, so corrected runtimes retain prior evidence
without colliding with it. Interrupted-window publication derives custody from
retained full checkpoint bytes; matching metadata files alone do not prove it.

Checkpoint reservation covers post-update metadata growth before SGD, including
the second asynchronous snapshot inode. A preallocated tensor payload alone is
insufficient: a 3,153-byte estimate increase exhausted admission below the free
space floor after an update. Large reservations include physically allocated
metadata headroom; growth beyond it still requires full admission. Never count
an update whose checkpoint failed to commit as durable progress. Retire abandoned
reserve inodes only after verifying the writer is terminal and the exact inode
contains no serialized checkpoint; preserve actual source weights and evidence.

Qualify the publication path alongside a source continuation, before spending
SGD on it. An authenticated target repair changes a row binding; preserve its
old exposure and quota, record the exact correction, and never claim unchanged
bindings. Completed SFT publication supports the typed correction custody proof
in `rl/native_direct_sft_recovery.py`. A carried SFT frame owns its own train/held
receipts and bindings; a surrounding RL frame describes different data. A consumed
source correction and its predecessor CLI argument retire together only after
authenticated central replay proves the correction already committed. An
uncommitted correction retains both strict inputs. Keep
manifest SHA256 and the combined trainer source identity distinct. Stage custody
preparation before a learner finishes, and cache verified extraction artifacts
so a corrected review script does not repeatedly decode full checkpoints.

Ordinary updates do not need to hash the rolling checkpoint for an oversize
capacity proof. When such a proof is needed, a concurrent atomic checkpoint
replacement requires a fresh checked read, with a bounded retry and writer drain;
never weaken file-identity checks. Distributed resumes bind the actual earned
rolling bytes while retaining the coordinator's original seed or exact allowed
role identity. Changing other fixed controls still requires its proper transition.

Native SFT/RL export current recovery states, not minimum-loss aliases. Loss
across changing data and objectives cannot qualify a production parent; avoid
duplicating full-state backup traffic for loss-selected aliases. Keep the
exported prompt-piece table synchronized with runtime guidance and retain
historical pieces for recorded-source rendering.

## Independent curriculum promotion and bounded discovery (engineering, 2026-11-01)

RL difficulty frontiers are per declared semantic task family. A solved math
level must not wait for unrelated weak tool/semantic tasks. Preserve rotating
breadth, least-covered-family exploration, actual task/sample budgets and complete
graph custody. Missing family metadata remains one unknown bucket. The resolved
policy records `frontier_scope: semantic-family`; historical frozen selectors and
bound collections retain their actual old policy. Apply new selection at the next
unbound collection; never rewrite an in-flight selected packet or old outcomes.

Read-only discovery must not become unattended I/O load. Scope searches to known
roots or declared source registries, bound broad searches, and poll or terminate
the exact returned process/session. Two orphaned root `find` searches ran6/13hours
across the external disk and dominated tiny random reads until stopped. They were
not training/conversion work. Use actual PID I/O to diagnose contention before
changing useful jobs or claiming a storage-throughput improvement.

## Useful data and recovery I/O custody (engineering, 2026-11-01)

Acquired exports without a D-number source label can share leaf directory names.
Their conversion output key includes the immutable source-path hash; a receipt
from another export must never count as a completed skip. Preserve conflicting
historical outputs, reconvert each real source separately, and inventory committed
receipts while other conversions continue. Receipt presence and output size are
not output-hash verification, current protection screening, semantic quality or
training admission. Keep source-scoped inventories explicit about their filter.

Memory-ledger I/O priority follows useful-work priority: high training/source
reads use best-effort0, medium preparation best-effort4, and low bulk conversion
idle class. The checkpoint exporter stays CPU nice19 but best-effort2 for I/O;
recovery must not wait behind normal-priority cold conversion. Apply priority
repairs to actual live PIDs and descendants as well as future launches. Preserve
the40GiB floor and independently verified parents while managing pending writes.

## Native numerical and observation custody (engineering, 2026-10-31 evening)

LFM3 normalization must share canonical row arithmetic in ordinary inference,
stored/reference execution and differentiable carriers, including forward,
gain-gradient products and input adjoints. A clean restore of corrected54 exposed
forward drift at layer20 despite identical inputs; backward-only fixes were
insufficient. The qualified native Torch path includes3cc2556be. SGLang is a
separate implementation: actor checkpoint attestation alone does not qualify
its numerical equivalence. Use the qualified engine while repairing faster ones.

Native CUDA BF16 matrix multiplication defaults to FP32 accumulation: reduced
BF16 reduction changed identical-prefix logits and hybrid state when row geometry
changed. Bind the actual reduction policy in actor and training identities;
`NATLANG_BF16_REDUCTION=reduced` is an explicit historical-replay override, not a
production default. The corrected checkpoint54 long forward/backward receipt
covers all288 trainable parameters; a numerical certificate is not capability
lift. Preserve reduced-arithmetic collections as historical data and never label
them current-policy samples after a numerical-policy transition.

Compile tensor kernels rather than unrolling Python invocation/chunk/norm
planning. Long native graphs exposed multi-minute Dynamo tracing before any
optimizer update. Keep default-on phase/stack diagnostics, and use the qualified
eager wrapper path when cold compilation costs more than useful execution.
Capacity qualification must exercise actual production packing, cotangent
backend, precision and retained-gradient geometry. Reuse its authenticated
geometry boundary across unchanged SGD execution; a new step alone is not a
reason to repeat a full backward for every microbatch. Distinguish that measured
capacity boundary from a fresh numerical certificate at the updated weights.

An infrastructure wall timeout without an accepted final answer is unresolved,
not a scored model failure or measured policy noise. Preserve raw observations
and exact re-scoring history. Deterministic model request/token exhaustion remains
an actual budget outcome. A protocol first-action match or mathematical gradient
certificate does not establish complete task success or capability improvement.

Keep staged-result notice construction and authenticated trajectory parsing in
`native/staged-result.ts` aligned. An exact old historical notice may remain a
supported input dialect, but a changed serving notice must not silently reject
valid current completions. Status-only success still needs the real same-invocation
staged action, exact output and terminal ledger custody. Task-level actual/expected
judgment alone does not certify helper contracts: inspect the whole execution
graph before admitting a teacher trajectory whose children have scoped duties.

Inspect the actual visible API contract before assigning task failures to the
model. The ToolACE converter discarded nested argument fields and enum values,
exposing `sequences: any` while the oracle required hidden exact object keys.
Preserve upstream schemas in typed service declarations and documentation; never
infer a missing public contract from gold answers. Requalify corrected tasks
through reference, negative and alias replay, keep held packets unchanged, and
do not relabel old underspecified-prompt rollouts as current-policy evidence.

## Machine-wide storage and bounded run state (owner, 2026-10-31)

Account for all project roots and shared stores, not just Natlang's data directory.
The owner authorized removal of `/home/werg/megatoken-runs` and aggressive cleanup
of `data/dgxops`; the MegaToken deletion is complete. Keep actual live source,
held-data and recovery dependencies, but historical directory names or hash
mentions alone do not justify keeping duplicate staging or orphaned code trees.

Use one rolling checkpoint slot per logical learner across restarts. Bind the
actual earned slot and prior controller parent through `rolling_sft_resume` under
the controller lock; never create another full checkpoint directory merely to
restart, silently ignore a conflicting explicit parent, or invent completed work.
Reserve the actual future pending inode before optimizer work. Stream into those
allocated blocks, preserve the durable parent until atomic publication, and retain
completed in-memory updates while a temporary storage shortage is repaired.
Budget independent exports, fallback states and concurrent staging explicitly;
an independently verified export is distinct from a local commit or upload ACK.

New bulk generation belongs on the verified external data volume, with its disk
floor checked on the actual output filesystem. Do not lower the internal free-space
floor to bypass pressure. Keep runtime snapshots and caches bounded, reuse qualified
dependencies, and stage the declared runtime/data closure before launch rather
than discovering missing ignored assets after allocation. Source and training
data survive cleanup; disposable unpacked restore tests and byte-identical packing
intermediates do not need permanent copies.

New frozen native runtimes seal their files into the adjacent content-addressed
`.native-runtime-file-pool` by default. Snapshot paths and ordinary tar exports
remain independent; identical read-only file inodes are shared. Edit main and
freeze a successor rather than modifying or chmod-writing a sealed file. Retire
unused snapshot directories normally; the pool prunes blobs only after no
snapshot links remain, under the publication lock. Preserve live runtime and
recovery bindings when choosing snapshots to retire.

## Actual early privileged learning and writer readiness (engineering, 2026-11-01)

Inspect `rl_sampling.privileged` and the converted objective/loss, not just
`sampling_law.guidance`: the latter describes import into the student. Early
backbone collection uses crisp selection and the verified text-twin fallback;
writer-off does not disable privileged self-distillation. Checkpoint57 earned a
real student OPSD update from six whole graphs and22 text-twin blocks, with zero
outcome/teacher terms. This is neither native-writer qualification nor demonstrated
task lift. Joint SFT's full native producer/consumer obligations remain active.

Current checkpoint56 native guidance writers all stalled on malformed producer
spans before solver work. Do not repeat that known-worse early configuration just
because mechanism tests pass. Repair and demonstrate actual authoring/consumption
before promoting to the later native-writer guidance stage; preserve exact failed
captures and keep useful text-twin learning and joint SFT advancing meanwhile.

Storage supervision follows current native writer processes and their reservation
receipts (`scripts/native_storage_supervision.py`), not an old learner PID/cursor.
It observes; the checkpoint writer enforces allocation. Preserve live
`checkpoint.pt.reserved-pending`, `.reservation.json` and `.reservation.lock`.

## Supervision and genuinely diverse Luna generation (owner, 2026-10-31)

Owner follow-up2026-11-01: subscription teacher generation is for genuinely
useful work. Up to six workers is a ceiling, not a utilization target. Bind
generation to specific missing Natlang/Neuralese behaviors, review complete
graphs and actual verification outcomes, and measure useful admitted yield
against requests/tokens. Repeated failures do not justify unchanged retries or
automatic expansion. Keep source conversion and on-policy local rollouts
distinct from paid external generation; preserve exact actor provenance.

Codex resumes supervision from `plans/neuralese/SUPERVISION_HANDOFF_20261031.md`.
Resume Luna dataset and trajectory generation with **up to six concurrent Luna
workers**. This supersedes the one-worker cap and the LFM3 prohibition on new
teacher generation below. Engineering and generation actors retain their actual
provenance; teacher trajectories are not on-policy student rollouts.

The owner explicitly requires **real diversity, not renaming**. New names,
domains, seeds, prompt views and shuffled records do not establish new task
families. Generate codebases with different computational problems, algorithms,
call graphs, state/effect semantics, interfaces and failure modes. Include actual
repository investigation, behavioral repair/implementation, evidence reasoning,
constraint planning and changing-state control. Prefer complete executable
applications and independently checked task outcomes over scaffold volume.

Measure repetition of normalized code, instructions, targets and graph structure
across packages and in the actual admitted mixture; retain semantic review since
syntax differences alone do not prove behavioral diversity. Loading a package is
not proof that a behavioral repair or implementation works. Preserve related-world
split custody, protected evaluation inputs, actual teacher identity, full execution
graphs and exact verifier evidence. Admit useful verified additions promptly;
never convert task counts, loader checks or teacher artifacts into capability claims.

Owner clarification: use diverse datasets to build substantial Natlang codebases,
then generate several different calls/tasks against each codebase. The reusable
application, its meaningful internal APIs and state, and its multiple operations
are the unit of generation. A generic dataset-answer wrapper is insufficient.

## Neuralese architecture replacement (owner, 2026-10-15)

The destination is shared positional KV and hybrid-state spans, replacing the
projected-value channel in BOTH training and inference. These instructions
supersede older projection/recursive graph-replay foundation rules for new work.

- Neuralese tokens use ordinary token embeddings with a learned low-rank
  transformation in embedding space. Do not train or collect a projection into
  a separate communication latent space in this phase.
- Producers supply positional KV spans and other shareable hidden state;
  consumers splice that state with correct positions. Handle LFM convolution
  and linear state explicitly. Train teacher-forced graphs in one graph forward
  pass using masks and state handling, rather than recursive graph replay.
  Verify outputs and gradients against explicit cached execution; do not claim
  one-pass execution while secretly replaying the graph.
- Use different beginning AND end markers for producing versus consuming
  Neuralese. Producers learn independently useful exported state; consumers
  learn to use imported state. Stored Neuralese includes text and a sequence
  of token embeddings. Preserve tokenizer provenance and make cross-tokenizer
  matching explicitly best effort, not assumed exact state interchange.
- Admit genuine producer/consumer communication graphs into the active
  trainer promptly; ordinary one-node text SFT alone does not train
  communication. Preserve completed per-row exposure and optimizer state.
- Replace the old approach end to end: shared trainers, conversions, inference,
  block storage/transport, natlang harness, evaluators and recipes. Retain all
  usable source data and protected evaluation splits through conversion.
- No prototyping shortcuts or half-solutions at architectural blockers. Work
  out both training and inference. After implementation, a separate agent must
  independently check the integrated work for gaps before cutover.

Track implementation and cutover in
`plans/neuralese/SHARED_STATE_MIGRATION_20261015.md`.

Complete producer targets (engineering finding, 2026-10-26): static pair
conversion must preserve the original complete assistant/tool response and its
declared result contract, placing producer markers only around the authenticated
exported span. Replacing that response with the payload alone taught incomplete
returns under unchanged typed prompts. Verify retained sibling fields, tool
wrappers and every consumer import on real source trajectories with both model
tokenizers. The affected converter family is quarantined until repaired sources
are admitted; preserve original source data and held groups. Historical exposure
to truncated targets does not qualify repaired communication foundations. Earn and record fresh exposure to the corrected row bindings while
preserving useful weights, optimizer/RNG and the original exposure history.

Task contracts and reward validity (engineering finding, 2026-10-26): an exact
historical next assistant message is an SFT label, not a valid execution reward
when other continuations can be correct and the recorded answer is not visible.
Use the authenticated task-source correction mechanism to quarantine these RL
rows without altering retained rows or losing source/split evidence. Admit exact
oracle-approved expected root stops with their actual invocation/action custody;
do not discard a correct blocked root merely because it is not `done`, or excuse
a failed child because its parent succeeded. Reference replay alone does not
qualify an oracle: test meaningful wrong answers and unmet required effects or
repairs too. Keyword/length constraints alone do not prove editing fidelity.

Publication dependencies (engineering finding, 2026-10-20): newly admitted
classification witnesses retain exact historical parser-policy dependencies.
Include their explicitly verified package roots in both provider and checkpoint
publication declarations. A shared provider fix does not update a frozen
trainer's publication helper. Check the actual privately verified step, not
service liveness or a saved checkpoint alone; keep useful training running with
the existing full-state fallback while repairing publication. Retain original
artifact hashes and exact path validation.

Exchange custody follows the effective shared contract (engineering finding,
2026-10-22): a runtime cutover can change role verification mechanism identities
and therefore the fixed exchange-cohort digest. Derive the complete effective
contract after all controller/publisher loaders and select the same fresh,
contract-addressed exchange directory before the real launch.
Do not reuse or overwrite a differently bound cohort. Rebuild only the completed
adaptation and control declarations when necessary; preserve earned weights,
optimizer/RNG/exposure and the existing training lineage. A failed coordinator
before learner launch is not evidence that its saved model state is bad.

Retention follows active lineages (engineering finding, 2026-10-20): a larger
step number from an old lineage does not supersede a current model's artifacts.
At every owner/runtime cutover, bind cloud retention to the current full-backup
and inference states, current evaluation consumers and in-flight publications.
Protect required current parents explicitly. Check actual remote presence after
publication; a successful GET can finish after a concurrent retention deletion.
An old readback receipt alone does not prove that recovery still exists. If a
current export disappears, stop the exact retention service, repair its active
dependencies and restore useful exports before resuming verified cleanup.

Native token learnability (engineering qualification, 2026-10-15): existing
tokenizer entries are not evidence of usable frozen output rows. The QAD
thinking boundaries were almost identical, imposing a fixed output-probability
limit regardless of backbone training. Current native models use learned sparse
rows for the existing reasoning boundaries and FP32 masters for all learned
Neuralese/reasoning rows, with the backbone's activation casts. Upgrade legacy
parents explicitly, preserving exact numeric parameter/moment values, RNG and
exposure; never reset history or silently reinterpret an old checkpoint.
Check actual parameter changes as well as gradients: BF16 sparse rows lost
production-rate updates despite nonzero gradients. The shared trainer records
per-row gradient/update norms and changed coordinates. Diagnose an unexpected
zero update before attributing poor learning to the model. See
`plans/neuralese/shared-state/REASONING_TOKEN_LEARNABILITY_20261015.md`.

Native optimizer routing (engineering qualification, 2026-10-16): native
ordinary parameters, including sparse vocabulary rows and the low-rank adapter,
use AdamW; the backbone's declared MuonSR routing remains separate. Do not
reclassify ordinary matrices into Muon by shape. Existing affected checkpoints
require the explicit `native-ordinary-to-adamw-new-history-v1` migration:
retain original numeric Muon buffers, groups and named bindings as transition
provenance, preserve weights/RNG/exposure and unaffected optimizer history, and
start the newly selected Adam algorithm with honestly new zero moments/age.
Never fabricate historic Adam variance or silently reinterpret optimizer kinds.
Monitor actual gold-boundary support, gradient/update directions and held task
behavior; a routing bug alone does not establish the cause of every behavioral
failure or an optimal learning rate. See
`plans/neuralese/shared-state/NATIVE_OPTIMIZER_ROUTING_TRANSITION_20261016.md`.

Owner correction, 2026-10-15: the lodestar is comprehensive training of BOTH
models through the ENTIRE production recipe and all usable available data.
Stages are steps to progress through, not endpoints. Half measures, short
probes and partial stages do not count as production completion. Track whole
recipe progress and actual data exposure; integrate newly prepared sources
through explicit continuations. KL distance from the base is telemetry only:
improvement requires changing the model. Do not stop or deny qualification
solely for base-relative KL or agreement drift. Keep actual task-performance,
held-loss, numerical and runtime checks; diagnose and repair real regressions.
Runs may be stopped and tweaked at any time when an improvement warrants it
(owner clarification, 2026-10-15). The supervising agent decides whether the
problem requires a redo or continued training, preserving exact state and
recording the evidence. Comprehensive training is the objective, not a ban on
interruptions or sensible course corrections.


## Adaptive production training: custody and continuation findings (2026-10-15..27)

- **Qualify current foundations on an explicit shared-stage request.** Retain
  genuine completed producer/consumer exposure. Repeated synchronous zero-update
  qualification must not replace full-source SGD between ordinary SFT/DPO/RL
  continuations. Publish a genuine qualified zero-update boundary through the
  exact normal mechanism without epoch, source-visit or optimizer-update credit;
  then continue training. Preserve unfinished full-recipe obligations.
- **Keep launch checks nonmutating and necessary** (engineering lesson, 2026-10-20).
  Run normal metadata/argument preflight once; let the real entry perform its
  required state, numerical and capacity checks before updates. Do not execute
  an entire administrative entry in the canonical production directory first:
  it can leave strict-write launch records that block the real learner. Put
  diagnostics in an isolated namespace, preserve concrete failure evidence,
  and avoid repeated whole-entry barriers while useful training is stopped.
- **Avoid redundant supervision barriers** (engineering practice, 2026-10-20).
  Routine data append payloads that change only counts, paths and authenticated
  generations may proceed through the already reviewed mechanism with an
  independent operator review and the normal schema, source/held, custody,
  geometry and compare-and-swap checks. They do not need another round trip to
  the root agent. Review mechanism changes separately. Record the actual
  reviewer and standing authorization; never claim an individual review that
  did not occur. Preserve exact exposure and continue useful training.
- **Bring in deferred improvements** (owner, 2026-10-16). Keep a current
  inventory of deferred data, performance, training and inference work; actively
  integrate each ready improvement. An old handover, incomplete source census,
  clean-run boundary or another model's unrelated qualification is not a reason
  to leave useful work disabled. Preserve real numerical, task-quality and
  foundation prerequisites, and resolve their concrete blockers.
  Independent model improvements may advance independently; propagate shared
  fixes and retain exact optimizer/RNG/exposure through adaptive continuations.
- **Share code as much as possible and distribute all generalizable insights
  across all models.** A useful fix, optimization, data improvement or recipe
  insight on one line must promptly be considered and propagated to the other
  lines and machines. Extend the shared trainers, loaders, objectives,
  evaluators and recipe inheritance instead of creating per-model forks.
  Keep only necessary architecture-specific behavior/settings separate, state
  why they differ, and communicate shared decisions through coordination.
- **Verify the effective control code after loaders.** A fix in the installed
  package can be overwritten by a sealed controller or publisher source closure.
  Rebase every control layer onto shared fixes, preserve required operational
  behavior, update mechanism identities, and test after the complete loader has
  run. A passing default-module test does not prove that the production
  controller executes the fix. This applies to both model lines.
- **Rebind capacity declarations after runtime changes** (engineering finding,
  2026-10-27). Even a checkpoint-control-only backport changes the exact Python
  inventory. Update the affected recipe and seed memory requests to the actual
  runtime while retaining source, objective, geometry and reference checks; let
  the real restored entry perform qualification before updates. Retain failed
  operation records and reserve an unused operation namespace when continuing
  from the latest earned full state. Do not silently skip comprehensive SFT
  because a resolved failure remains memoized under an unchanged signature.
- **Keep recovered work ahead of historical operations.** Restarting the full
  mandatory schedule after an external optimizer continuation must reserve an
  unused operation namespace in the same lineage. Never let a completed bridge
  result from an older attempt replace the newer earned parent. Verify the
  actual checkpoint and cursor, and observe the real optimizer child advancing;
  a live coordinator or a launched request alone is not resumed training.
- **Reuse actual controller custody across continuations** (engineering finding,
  2026-10-21). Pass the coordinator's continuously held canonical lease descriptor
  through worker threads into completed/runtime configuration; opening the same
  lock again can self-contend. A normal resident publication changes bootstrap
  metadata: verify and refresh the actual adopted configuration rather than
  discarding its data or weakening custody checks. Preserve concrete configuration
  tracebacks and diagnose a blocked retry signature. Exercise this path after the
  complete sealed controller/publisher loaders; a previously shared guard fix can
  still be missing from their frozen source closures.
- **Publish completed resident training before adopting its next stage**
  (engineering finding, 2026-10-22). A supervised stage can finish after several
  live data appends while the coordinator still holds its bootstrap recipe.
  Use the real completed request and result through the normal publication
  helper to bind the earned checkpoint, actual resident configuration and next
  cursor. Updating bootstrap metadata alone does not publish the adopted recipe.
  Preserve optimizer/RNG/exposure and verify the next real optimizer child.

Historical reader replay (engineering finding, 2026-10-22): recipe inputs can
include the exact classification reader implementation. A changed reader is not
a byte-identical relocation. Replay historical completed adaptations and drained
handoffs using the package that actually prepared their metadata, with complete
code inventory and bound recipe checks before and after isolated CPU replay.
Both old and future plans in a completed adaptation were enumerated by that
adaptation's implementation; its old numerical package is not automatically the
original metadata reader. Preserve strict source/held, lineage and exposure
checks. Include committed resident histories and both effective sealed control
closures in validation before cutover. A live scheduler holding the canonical
lease can still be blocked: monitor its actual configuration_errors and retry
signature alongside real optimizer progress. When a normal publication advances
the earned parent, rebind a completed invocation to that actual state through
the existing continuation mechanism rather than reusing an obsolete invocation.


## Working practice: launch, receipt and copy findings (2026-10-22..11-01)

- A ledger unit's explicit relative executable path resolves in its admitted
  `--workdir`. Bare executable names retain PATH lookup. Use an absolute script
  path for an intended checkout when inspecting or recovering older launches:
  older `systemd-run` invocation paths resolved against the caller's checkout
  before changing directory. Verify the tested checkout's path and revision in
  the actual check log; a worktree claim alone is not evidence of its tests.
- For DGX administrative ownership guards requiring `os.pidfd_open`, use the
  host `/usr/bin/python3`; the CoreWeave SDK venv lacks that API. Invoke SDK
  operations through its venv as a subprocess when both capabilities are needed.
  Compile the assembled remote Python body before transport, not only the
  outer launcher. Honor each receipt's binding schema: compare path/SHA fields
  explicitly and check declared byte counts against the actual file; a legacy
  two-field binding is not unequal evidence merely because another includes bytes.
- Reuse authenticated live parser/index caches for CPU preparation and isolated
  consumer checks. Cache isolation is not an objective. Verify exact source,
  tokenizer, role, parser-policy and entry identities before reuse; preserve
  cold work only when no valid completed cache exists. Locate artifacts through
  registries and bound manifests; do not leave unbounded content searches or
  directory crawls running across the external disk after the answer is known.
- Bound dirty file-cache memory during large artifact copies (engineering finding,
  2026-10-22): a small Python heap does not bound a ledger unit's memory. Flush
  and synchronize bounded write windows, release clean cache when appropriate,
  and resume an interrupted copy only after validating its actual prefix. Verify
  full content before replacing a logical path; preserve source and live readers.
- Do not add or remove hardlinks to source, manifest or index files while live
  native readers validate or stream them. Link-count changes alter ctime even
  when bytes remain unchanged and correctly invalidate strict reader checks.
  Copy for live backup staging or drain the affected readers first; keep the
  strict checks. Immutable checkpoint pinning has its own checked-stat protocol.
- Restart transient ledger services through the ledger with a fresh unit name
  (engineering finding, 2026-10-22). A stopped transient unit may already have
  been collected, so `systemctl start` can fail even after a clean stop. Preserve
  the existing state/lease namespace, verify the new actual process and update
  supervision bindings; never treat scheduling the wrapper as resumed work.


# Moved from AGENTS.md 2026-11-15

The 2026-11-15 cleanup of `AGENTS.md` kept only what an agent needs to act today. The text below is the pre-cleanup
AGENTS.md (commit b3b6543b7), verbatim, for every section that was condensed, rewritten or moved out: run histories and
smoke numbers, the full GRPO-loop description, held-v2 details, HF fast-train benchmarks, the retired w5 machinery, the
platform and coordination history, and the dated owner statements that were thinking steps rather than standing rules
(owner 2026-11-15: "What you misunderstood as rules were just steps along the way to think through what is best"). The
natlang-program-writers note, the natlang-native product rules and the shared-machine working practice stayed in
AGENTS.md with light trimming and are not repeated here. Headings are demoted one level.

## How we train now: bounded controlled experiments (owner, 2026-11-12)

`plans/neuralese/APPROACH_REVIEW_20261111.md` is the plan; these are the standing rules for model training. No w5
checkpoint beats the untouched base on held-out (base 5ff07638 54/118, step 92 50/127, steps 110-134 down to 39/128),
so w5 is retired (data kept, never resumed) and the base is the start policy.

- **Precision.** Fine-tuning keeps FP32 master weights and FP32 optimizer state and computes in BF16
  (`scripts/neuralese_data/exp0_sft.py`). Pure-BF16 parameters with stochastic-rounding writes are not used for
  LFM3: at our learning rates those updates were rounding noise (review section 1.1).
- **Bounded controlled experiments decide.** Each experiment is one collect -> train -> eval cycle on one pinned
  commit (actor, harness, scorer, learner and evaluator all run it; record the SHA), changes exactly one thing
  against its baseline, declares its success criterion before it runs, and is judged by paired held-out eval.
  Experiments are small (hours). Record each with its result in `plans/neuralese/DECISIONS.md`.
- **Improvements enter at the next experiment, never mid-measurement.** Code landing on main does not reach a
  running experiment: a pinned run uses its own copy of its commit (a detached checkout or a pinned copy of
  `ts-host`; no commits are made there, and it is removed afterwards). A measured win becomes the baseline of the
  following experiments, with an opt-out switch. A correctness bug found in a running experiment invalidates it:
  stop it, fix it, rerun it.
- **Learner data path: per-call sequences.** Each model call of an episode is one (prompt, completion) sequence. The
  rendered prompt is zero-weight context, the sampled completion is the target, and the episode reward is the
  weight. Experiment 0 is SFT on own successes, Experiment 1 is a rejection-sampling fine-tune (256 tasks x 8; pass
  is >= +8 held successes, p < 0.05), then plain GRPO (group 8, per-call sequences, KL to the start policy).
  Whole-graph Neuralese gradients return as a controlled addition once there is Neuralese capability to train (use
  rate today 5-13%).
- **Off until each returns as its own controlled addition with a paired measurement:** the JOLT+ teacher and
  privileged twin, the Neuralese bonus, memetic guidance programs, expert-iteration window shares, Pop
  pre-collection, interleaved SFT windows and QAT. They come back one at a time, after two consecutive held gains
  from plain updates. Neuralese-declaring task variants come first, because they are data, not machinery.
- **Evaluation gets real compute.** Paired held-out eval on a large pinned set decides every comparison (held-v2
  below; mean per-task pass-rate change with a within-task permutation test and task-bootstrap CI, timeouts as failures;
  McNemar, task sign test and loop rate beside it). dev16 is a smoke test at most,
  never a gate.
- **Heartbeats check, they do not re-plan.** A monitoring cycle checks that the pinned experiments run correctly and
  the machines are healthy, and fixes breakage. Plans change between experiments, from their results.
- **A significant unexplained held regression ends the experiment at that eval** (owner, 2026-11-14). A paired held
  result below the declared failure threshold with p < 0.05 whose cause is not known stops the run and starts the bug hunt
  on the freed machine. The declared failure rule is a last resort, not permission to continue: grpo-v2-run1 ran 7 more steps
  (8 h, ~100 GB of the DGX) after step 8's -8.1 pts until step 12 confirmed it, while learner checks needing ~10 GB waited.
  Likewise, a learner sentinel firing (grad_norm >5x the run's median, a reasoning-length shift, KL jump) is diagnosed at
  that step, not watched. "Don't touch a pinned experiment mid-measurement" protects valid measurements; a regressing run
  with a suspected bug is not one.
- **Each heartbeat asks what the machines are worth.** Is any running job's result already decided or uninformative, and is
  anything (diagnosis, the next run) waiting on memory or GPU it holds? Then stop it. Every run leaves >= 15 GB of DGX
  headroom for diagnostics and probes. A run's stop also stops its Pop eval actor. The heartbeat prompt names the
  experiments actually in flight; a stale list is fixed in the same cycle.
- **Diagnostics are small, fast and branchable** (owner, 2026-11-14). Insight came from minute-scale checks (probes, offline
  statistics on archived rollouts, grafting), while full-size replays took hours each to confirm them. Hence:
  - Every run self-diagnoses: the behaviour probe runs after each step (`behaviour` in `train.jsonl`: sampled-token logp
    change and entropy on 48 calls the start policy sampled, `</think>` hazard; `behaviour_shift` flags it; on run1's
    checkpoints it stays quiet at step 2 and flags step 8 (8.9x the closes) and step 12 (entropy +0.075)). The held eval decides comparisons;
    it is not how a drift is first seen.
  - Runs are branchable: `--keep-checkpoints-until N` keeps every early checkpoint, `--branch-states N` the last N FP32
    states (about 21 GB each, on the checkpoint disk); an ablation resumes from the step it questions (`--resume
    --resume-state FILE`, with `--replay-schedule` for the batches after it) instead of re-running from the base.
  - Learner questions go to the learner bench first, in minutes: `grpo.py train --replay ROLLOUTS --replay-max-group-tokens
    16000 --min-groups 6 --max-groups 6 --max-steps 6 --keep-checkpoints-until 6` (short archived groups; with
    `--advantage-control shuffle` a learner bias shows as a probe shift; `lr_data_scale` keeps small steps small). Offline
    statistics of the estimator on archived groups come before any GPU run. Full-size runs answer only what the bench cannot
    (on-policy feedback, held outcomes).
  - Diagnostic units are small: one question per ledger unit, the fewest steps and groups that answer it, results after
    each step rather than at the end.
  - Code analysis starts early, in parallel (owner, 2026-11-15): as soon as a defect is suspected (a sentinel fires, a
    held regression, a surprising number), start a static analysis of the code path in parallel with the experiments,
    without waiting for them to localize it. It needs no GPU, so it costs no machine time, and it serves as the
    independently briefed second investigation. Brief it with the observations, not a hypothesis, and pass it each new
    experimental result as it lands. On grpo-v2-run2 the analysis started only after hours of experiments had narrowed the
    defect to the learner.
  - Fastest route to the answer (owner, 2026-11-15): before any diagnostic or sweep starts, its brief states the question,
    the wall time each candidate design will take (steps x seconds per step x arms), and whether each design reaches the
    regime it tests (warmup vs step count, LR actually reached, sample size); a design that cannot answer is not run. Then
    choose by time to a trustworthy answer, not by cost: the methods are offline statistics on archived data, forward-only
    evaluation of weights that already exist (scaling an update `base + a*(theta_k - base)` as a stand-in for a learning
    rate, grafting, interpolation, sign-randomized nulls), short learner-bench replays, and full-size runs. If the slow
    method is likely to be decisive where the fast one is not, run the slow one. If a fast method may suffice but the slow
    one may still be needed, start both in parallel when the machines allow, and stop the slow one as soon as the fast one
    answers. The supervisor checks this for every design it writes or approves. On 2026-11-15 an LR sweep of five 2 h
    full-size arms was launched whose warmup-10 arms could not reach their learning rate in 5 steps; a forward-only scan
    along an existing update answered the same question in about an hour.
- **Pop runs held evals first and diagnostics when none is queued** (owner, 2026-11-14): a probe or check
  may use Pop's idle GPU when it fits (RTX 4060, 8 GB; 14 GB host RAM: probes and forward-only checks of a 1.7B BF16
  model, not the learner); a held eval of a pinned run preempts it.
- **The training and RL stack uses standard tooling** (an HF training loop, the SGLang server) and minimal crisp
  code. Build infrastructure only when the next experiment needs it; no new transitions, qualification scopes,
  controllers or custody schemas.
- Agents adopt plans like this one themselves: waiting for the owner's OK on a technical course change is a defect
  (owner, 2026-11-12).

### GRPO loop: plain GRPO from the base, overlapped (engineering, 2026-11-12)

`scripts/neuralese_data/grpo.py train` is the review's "Hour 24-48" GRPO (a few hundred lines; no controller, custody or
transitions). One ledger unit holds the learner and, as a child process, its training actor (held-v2's pinned engine:
SGLang vendor projection, FP8 KV; a separately admitted actor unit deadlocked behind the learner's claim). Collection never stops: `--slots` work-stealing episodes over an endless
shuffled stream of task groups (8 samples; pools: the w5 pool plus the verified bgkit KR rows, fallow K1; held-v2 excluded
with its split groups and every row sharing a source group; `WORK/pool.json` records the pool shas). The learner trains on
whatever groups finished since its last step (16-128; stragglers join a later step, groups more than 2 versions stale are
dropped). Tasks (EPISODE_COST_20261112): 85% from tasks whose measured base pass rate is in [0.1, 0.9] (exp1-rft
`base-pass-rates.json`, re-read every 10 min; a task's own latest scored GRPO group replaces its rate, so tasks the policy
solves 0/8 or 8/8 leave the band and explored ones enter it; exp1's pass-rate collection stopped 2026-11-13), the rest unmeasured pool tasks (`--explore`, default 0.4 from 2026-11-13, run2 runs 0.15: below ~0.35 the learnable set drains, and family/level priors do not predict which task is mid-range, TASK_SUPPLY_20261113); Neuralese-declaring variants `--neuralese-variants` are off until their controlled addition; `ext_codebase-{py,natlang}_codebase_*` excluded
(repos missing on the DGX). Episode budget 40 model calls (a budget stop scores 0; the collector has no generated-token
budget yet), wall clock 3600 s as a safety net only (unscored; run2: 4 of 1,100 episodes, all model decode, no tool time). An episode killed by a signal (unit stop, OOM killer) abandons its group, which is never scored (run2's restarts had scored 127 killed episodes as `unresolved`, leaving groups biased to short episodes). Reward = task outcome only: by default graded (`--reward graded`, opt-out `binary`): 1 accepted; an episode whose only failed checks are the answer and the files gets the share of the files it had to write (halved when the answer is wrong too) that it wrote right, with stray changes and unremoved files as failures and nothing for doing nothing (`task_rollouts.graded_reward`); selection pass rates and held eval stay binary. Per-call sequences with the actor's exact prompt and sampled ids
(`lower_rollout_turns.episode_turns`, chained as in `exp0_sft.chains_of`); sampled decision calls train as one action each
(`decision_turns`: log softmax of the learner's option scores over prompt + each captured continuation); Neuralese-state
consumer calls train too, the imported state fixed context spliced from the actor's block store as the fork fed it (K/V
pre-RoPE, conv B*x; `lfm3_fast_train.import_tensors`, no producer gradient; `plans/MODEL_FACING_CHANGES.md` 2026-11-13); loss on sampled tokens
`-min(rho, 2) * A * logp + 0.02 * k2(start || policy)` (k2 = d²/2 with d = logp_start - logp, `--kl-estimator k2`; the k3 loss e^d - d - 1 gave a few T=1 tail tokens at d ≈ 10-12 ~90% of grpo-v2-run1 step 7's clipped update; k3 stays the logged `kl_mean`, beside `ref_dlogp_max` and `ref_dlogp_over_5`), `A` group-normalized, summed over all the step's units and divided by their count (`--loss-norm token`, from 2026-11-14:
`episode`, averaging over each episode's own units first, is the score-function estimator of E[A/T] with T sampled; successes
make fewer calls, so it rewarded shorter reasoning in every call, which the task never selected: run1's groups z +8.8, a candidate
cause of grpo-v2-run1's drift under replay test, DECISIONS 2026-11-14), a sampled
decision counting as one average call of its episode, at most 8 token units (`--decision-weight call`, `--decision-unit-cap`: at a
whole call's ~1,000 units one step reversed templated inbox decisions by ~18 nats), no policy term on a token the actor sampled below
log-probability -7 when `A > 0` (`--tail-logp-floor`; 0.5% of tokens at untruncated T=1). Sampling law (owner 2026-11-13): temperature 1 with
top-p 0.95 for collection, held eval and the learner alike (`common/sampling.py` `RL_TOP_P`, `--top-p`, 1.0 off; it travels as `top_p` in
`sampling-config.json` and each capture's `sampling_law`; the front refuses a reply it did not sample that way). The actor records the
untruncated logp; the policy term scores the token under the learner's own nucleus (`lfm3_fast_train.linear_nll_nucleus`, the sampled token always
in the set), `rho` and the KL term use untruncated logps (`rho` takes the actor's nucleus mass as the learner's: |d log rho| <= 0.051); calls of
another law are refused (`skip_other_sampling_law`); decisions keep their full option softmax; FP32-master AdamW, LR 1e-5 (check 4; run2 at 3e-6:
half of the BF16 flips reverted, DECISIONS 2026-11-13) times the step's groups with reward spread over 12, at most 1
(`lr_data_scale`, `--lr-reference-spread`: AdamW's step does not shrink with the data behind it; run1's 2-group stop step 3 took a
full step and the drift entered after it), BF16 compute
(`exp0_sft.load_model`, `lfm3_fast_train`). Weight sync each step: BF16 weights -> actor's HF safetensors on /dev/shm (memory-mapped and page-locked: the only host copy) ->
SGLang `pause_generation(retract)` / `update_weights_from_disk` / `continue_generation` in a background thread; a pruned
native checkpoint every 2 steps and the FP32 state (masters + moments, `/mnt/external/natlang-grpo-v1`) every 4 (grpo-v1 runs `--state-every 1`: a guard stop loses no step), also in
background threads. The whole run executes from its pinned copy (`scripts/pin_checkout.py create COMMIT
WORK/pin-<c12>`: detached worktree, ts-host rebuilt there; launch that copy's grpo.py with it as workdir; grpo.py refuses a
live checkout unless `--unpinned`), and every child (actor, episodes, held_eval measure/compare) gets its PYTHONPATH, so
held results record the pin's commit and series; adopting newer code is a new pin and `--resume` (when the series changes, grpo.py moves `held/step-00000` to `step-00000-series-<old>`, measures the start policy in place on the actor, restores its weights and measures the current step at once). After each step it deletes actor-store state blocks untouched since 10 min before the oldest outstanding group was created (`collect_blocks`; the store touches a block on every read and repeated write; `--keep-rollouts` keeps all). Held evals run on the training actor itself (it serves the
current policy with the held-v2 engine pin): new collection pauses, `held_eval measure` (held-v2; 4 samples for the start
policy before the first step, then 2 samples every 4 h on that step's checkpoint) and `held_eval compare` with step 0
(`WORK/held/step-N/`); `--held-extra CKPT,...` measures other native checkpoints the same way after step 0 (loaded into the
actor in place, `held/extra-<sha12>/` with a compare against step 0, 2 samples by default), then restores the start policy. `--held-remote pop` (owner throughput decision 2026-11-13; run2 uses it) moves the
held evals to Pop's GPU and the DGX never pauses: the start policy once as `held/step-00000-pop` on Pop's engine (compare
pairs only one engine), then the newest native checkpoint every `--held-every-hours` or back to back if an eval takes
longer (`pop_rl_actor.py serve-checkpoint`: rsync of the pruned checkpoint, the actor runs a `git archive` of the run's
commit on Pop, Pop keeps 2 checkpoints), measured by this run's pinned harness from the DGX against Pop's endpoint
(`--held-remote-slots` 14, `held-pop.log`). Pop's engine is KV-pool bound (2026-11-13): BF16 KV held 94.7k tokens, so 7.5 of 10 sequences ran, waiting requests evicted the prefixes of episodes between calls (prefix hit 41% vs 94% on the DGX) and decode tokens/s scale with running sequences (~20 ms a step at any batch); hence FP8 KV and 16 running (`--pop-kv-cache-dtype`, `--pop-max-running`; FP8 is part of the attested engine, so a start-policy result of another setting moves to `step-00000-pop-<time>-other-engine` and is re-measured). Step sizing (smoke): ~27k forward tokens per episode (chaining already merges 282 of 329
consecutive calls; 38 diverge inside the sampled reply, 9 in the prompt), learner 600-800 tok/s while sharing the GPU with
decode at `--activation-gb 4`; defaults now 24-48 groups per step (12 gave about 75 informative episodes a step), `--activation-gb 12`, 8 collector slots during the learner phase
(`--learner-phase-slots`: check 4 measured the learner 3x faster with actor decode at 32 slots than at 48-64; the run logs
actor tok/s per phase). The KL
reference runs the policy's own training graph (a no_grad pass compiles other kernels: KL 5e-4 at identical weights). Both passes compute a prompt prefix that several of the step's chains share (>= 256 tokens: the system prompt, a group's opening prompt, an episode's earlier calls) once, as a prefix tree (`lfm3_fast_train.prefix_tree`/`run_prefix_tree`: the prefix's keys/values and conv tail feed its continuations, whose gradients reach it through detached copies and flow back through it once, exact; prefixes are fully recomputed and come out of the leaves' `--activation-gb`; chains reading imported state stay whole; `--share-prefix-tokens`, 0 off). On run2's groups it removes 35-50% of forward tokens; `train.jsonl` logs `chain_tokens` and `shared_prefix_tokens_saved`. After each in-place update the front attests what it serves (`POST /v1/neuralese/rebind-policy`:
the step's checkpoint sha, else the served weights' sha; the native engine keeps the start checkpoint). `train.jsonl` logs per step: reward
mean, spread fraction, KL, rho (mean, pre/post-cap max, capped fraction, |log rho|), the cosine of the step's gradient with AdamW's first moment
(`grad_moment_cosine`), the pre-clip grad_norm over the run's median (`grad_spike` above 5x from the second step, printed: grpo-v2-run1 step 7 was 11x),
the behaviour probe (`behaviour_probe.py`, `--behaviour-probe`, default `ops/grpo-v2/behaviour-probe-v1.json`: 48 calls the base
sampled in run2, selected by the served export's sha because a capture's `policy` keeps naming the start checkpoint after
in-place updates (v0, 44 of its 48 calls from updated policies, is what grpo-v2-run1/run2 and the hunt measured); ~42 s a step), `--call-fraction` (default 1: a fraction of each episode's calls trained, the rest context), the gradient-mass share of reasoning, reply and decision terms, the tail-masked fraction, learner and actor tokens/s, phase seconds.
Memory (smoke 2026-11-12): on the GB10 the unit's MemoryMax (= ledger budget) also bounds its CUDA allocations, so the
budget must cover GPU + host: learner CUDA 19.3 GB before AdamW state, 32 GB peak at step 1, about 36 GB with the FP32 moments
(FP32 masters + moments + FP32 gradients alone are 27 GB), host about 5 GB plus the collector node processes (in the same
cgroup, 0.3-0.7 GB per slot; shared collectors save about 0.25 GB a slot), plus the child actor (kv-gb + 24). From 2026-11-15 (after run2's guard stops at 92.6 and 96.7 GB): the KL reference is a file
(`ReferenceWeights`, state dir) loaded into the policy's own parameters for the reference pass instead of a resident copy
(-3.5 GB), FP32 gradients exist only from the policy pass to the optimizer step (-6.5 GB outside it), imported states are
read per sequence instead of cached through the learner phase, the nucleus loss casts inside its kernel (-1 GB at the loss),
no pinned weight buffer or checkpoint clones (-4 GB, -4 GB transient); `train.jsonl` `cuda_peak_by_phase_gb` gives each
learner phase's CUDA peak with this process's host memory. The actor's
weight file lives on /dev/shm (the shared NVMe wrote at 60-230 MB/s under load); pruned checkpoints go to /mnt/external
(last 2 plus held-evaluated ones), the single FP32 training state (20.7 GB, temp file renamed over the previous) to NVMe.

### Held-v3: the experiments' held-out set for experiments pinned after grpo-v2-run1 (engineering, 2026-11-13)

`ops/held-v3/manifest.json` (sha 45471060; `plans/neuralese/HELD_V3_20261113.md`) is held-v2 with 14 tasks replaced:
wrong or ambiguous gold, and source groups shared with Exp1 training. It uses the same engine.json and is
`grpo.HELD`. Its `exclusion_ids()` also covers held-v2 (`also_excludes`), so neither set's tasks or siblings ever
enter training. Before relying on it, rerun `held_eval verify` against `ops/held-v3/exposure/collect_exposure.py`.
Results on it use the 2026-11-13 scorer (QA answer cores, terminal-newline policy, loop evidence for iterateOn
markers): a new series. The held-v2 section below describes the predecessor and the shared tooling.

### Held-v2: the experiments' held-out set (engineering, 2026-11-12)

`/home/werg/data/lfm3-rl-dgx-20261029/ops/held-v2/manifest.json` (sha dfe0a915): 256 pool tasks (the w5 pool b3fef4c7), 91
families, levels 63/64/64/65, 85 effect-state, keeping w5's 64 held tasks; disjoint from Experiment 1's selection, Experiment
0's training tasks, w1-w5 attempts and their `split_group` siblings. Training code excludes it with
`held_eval.exclusion_ids(manifest)` (held ids plus siblings, 1,170 ids; also `held-v2/exclusion-ids.json`). Never train on
these ids. Built by `python -m natlang_neuralese.rl.held_eval pin` (`--include-manifest`, `--avoid-ids`, `--avoid-from`).
Measure with `held_eval measure --manifest M --out DIR (--endpoint URL[=SLOTS] | --checkpoint CKPT) [--samples 4]`
(work-stealing episodes, resumable, sample-major so a cut run stays balanced; `serve` launches/reuses a ledger SGLang actor) and
compare with `held_eval compare A/result.json B/result.json [--ignore-ids F.jsonl] [--trained-families F.json]` (primary:
mean per-task pass-rate change, within-task permutation p, task-bootstrap CI, a wall-clock timeout counted as a failure;
beside it the drop-timeouts sensitivity, McNemar on matched samples, task sign test, loop rate; subsets without the
`held-v2/ignore-ids-*.jsonl` tasks and by trained family). A timed-out episode stays unscored in `result.json`
(`timeouts`); compare only results with one code commit and series (`same_code`, `same_series`; the series covers the
ts-host tree and the Python scoring sources) and one sampling law (`result.json` `sampling`: temperature 1, top-p 0.95 from 2026-11-13,
`measure --top-p`; `compare` refuses to pair laws, so the start policy is re-measured at the next pin). Training and held eval share one call budget (`EPISODE_MODEL_CALLS`, 40). Baselines and the A/A noise floor: `ops/held-v2/runs/`. The eval engine is
pinned in `held-v2/engine.json` (SGLang, vendor projection, FP8 KV, `--sglang-kv-gb 16`, 128 running; one engine per
checkpoint): `measure` refuses another actor (`--allow-other-engine`), `serve` launches exactly it, and `compare` refuses to
pair results with different attested engines (`serving_engine`, export format). Interim A/A (base twice, independent seeds,
shared actor, 1 sample on 56 matched tasks): 10 discordant pairs (18%), split 8:2, delta -10.7 pts, task-bootstrap CI
[-21, 0], p=0.11: one-sample differences under ~8 pts at 128 pairs are noise; read held-v2 at 4 samples (~3 pts at 1,024
pairs). Episodes take ~15 min at the shared actor's load and ~250 MB host memory each (budget the ledger unit for slots).

### GB10 CUDA contexts fail under memory fragmentation (engineering, 2026-11-12)

`CUDA_ERROR_OUT_OF_MEMORY` from `cuDevicePrimaryCtxRetain` with tens of GB free (kernel: `NVRM ... NV_ERR_NO_MEMORY ...
kgrctxAllocMainCtxBuffer`) is fragmentation: a context's buffers need physically contiguous pages, and the Normal zone had
almost no free order >= 8 blocks (`/proc/buddyinfo`). Measured: context creation failed whenever order-8 blocks were ~3,
succeeded at 12+, and 30/30 with the compactor running. `natlang-gb10-compactor` (persistent user service,
`scripts/systemd/`) faults THP-advised memory whenever under 512 MB is free in such blocks (direct compaction, no root);
keep it running. `common/cuda_context.py`: `ensure_context()` (serve_native_checkpoint, grpo, exp0_sft) and
`start_sglang` retry a context failure 3x after a compaction nudge and 20 s. Root fix (owner): `vm.compaction_proactiveness`.

### RL actor on the DGX: one engine, many slots, absolute KV size (engineering, 2026-11-12)

Collection decode on the GB10 is memory-bandwidth bound: each step reads the weights (~3.5 GB) plus every running
sequence's full-attention KV (12 KB per context token BF16, 6 KB FP8; RL prompts average 11.5k tokens). Exp0/w5 actors
ran 16-38 slots, 30 running sequences, pool 93% full with queued requests, 8-14 tok/s per stream. Rules:
- Run ONE SGLang engine per GPU and give it all slots (exp1/held/GRPO share it): two engines time-slice the GB10 and
  every kernel waits for the other's slice. Serve with `serve_native_checkpoint.py --engine sglang` defaults:
  128 running, `--sglang-kv-gb 12` (pool in GB; tokens and SGLang's mem fraction are derived from the memory free at
  launch, so other jobs neither shrink the pool nor get grabbed). Ledger budget = `--sglang-kv-gb` + 24 GB (the front reloads the native model for write_text).
  Do not pass `--sglang-mem-fraction` (SGLang reserves (1 - fraction) of free memory: 0.25 left a few GB of KV).
- Collector slots >= 64 per DGX actor; measure with `held_eval measure`. A slot costs ~250 MB of host memory in its own
  node + python processes; episodes share node processes, 8 each, by default (`cli.js --serve`; `NATLANG_COLLECTOR_SHARED=0` opts out).
  Details and numbers: `plans/neuralese/ACTOR_THROUGHPUT_20261112.md`.
- Front host memory (2026-11-15/16). Three growths of `serve_native_checkpoint.py`'s front, each bounded in turn: the
  block store keeps a 2 GiB resident LRU (c06a50581); glibc arenas are capped and trimmed every minute and the
  append-only token history is byte-bounded (ef5e09be7, 3d3f00c45); and torch's CPU tensors no longer pile up in idle
  threads. torch on aarch64 allocates CPU tensors with mimalloc (static in libc10; its arena is the 1-2 GiB
  `MAP_NORESERVE` + `MADV_HUGEPAGE` mapping in `/proc/PID/smaps`), whose heaps are per thread: an idle thread keeps
  the pages of tensors it allocated, including those freed by other threads, until it allocates again. The front ran
  SGLang requests on a 512-worker `ThreadPoolExecutor` that spreads work over every idle worker: grpo-v3-run1 front
  4.6 -> 12.4 GB in 12.5 h (~0.8 GB/h); in r1, ~180 workers for 32 connections after 30 min with 85% of anon growth in
  torch's arena. Synthetic: 128 round-robin workers > 3.7 GB (killed at 4 GB) for a 0.25 GB tensor LRU, a thread per
  task flat at 0.7 GB, glibc/numpy allocations with 128 workers flat at 2 GB. Each request now runs on its own
  thread that ends with it (`SglangEngine._spawn`; score fan-out `_map_threads`). `MIMALLOC_PURGE_DELAY=0` does not
  help. Any long-lived thread pool that allocates torch CPU tensors has this growth.

### HF-path fine-tuning speed: elementwise and dense sliding attention, not matmul (engineering, 2026-11-12)

`exp0_sft.py train` ran 1.8-1.9k fwd+bwd tokens/s on an idle GB10 (11k-token chains). Profile: matmuls (nvjet) 1.4 of 5.1 GPU s at about 90 TFLOPs; unfused
elementwise about 2.5 s; sliding layers as a dense masked 11k x 11k mem-efficient kernel 0.9 s (window 256); FP32-master accumulate + AdamW + copy-back 2%.
`scripts/neuralese_data/lfm3_fast_train.py` (reads the HF instance; vendored sources stay hash-pinned): `prepare(model)`, `hidden(model, ids, lengths=None)`,
`linear_nll(h, weight, targets)`, `token_logprobs(model, ids, positions, targets, lengths=None)` (no grad: GRPO behaviour/reference pass). The decoder layer is two
functions over explicit tensors (conv, attention), each `torch.compile(fullgraph, dynamic)` once and shared by all layers of its kind; attention is varlen flash
(`torch.nn.attention.varlen`, sliding layers window (255, 0), exact); `lengths` packs documents into one row (positions restart, attention and conv taps stay
in the document); the conv is seven fp32 shifted adds; `linear_nll` keeps only the fp32 log-sum-exp and recomputes each 4096-row logits chunk once in backward
(the stock checkpointed CE ran the head 4x and stored fp32 logits). Rounding points follow HF. `exp0_sft`: `token_nll` uses `linear_nll` on prepared models;
`set_checkpointing` recomputes layers beyond `--activation-gb` at 88 KB saved per token-layer (measured 84.8); a step's chains are packed into rows up to
`resident_tokens` (the longest row the budget holds unrecomputed, about 10k at 20 GB; `--pack-tokens`, 0 off): SFT v2 chains are short (median 2.0k).
Measured (2026-11-12, GPU shared with two SGLang eval actors at 94-96% util, so absolute rates are about a third of idle; A/B alternated back to back):
fwd+bwd all layers recomputed 1370 +- 39 (ed2ac83c0) -> 1471 +- 26 (new layers) -> 1496 +- 8 tok/s (+ fused CE), GPU s/chain 9.04 -> 8.36 -> 7.90;
packing on 20 SFT v2 chains 1739 +- 29 -> 1773 +- 20 (4k rows) -> 1803 +- 22 (8k); no-grad logprob pass 2982 +- 366 (stock HF) -> 7330 +- 1270.
Agreement: new vs ed2ac83c0 gradient cosine 0.998 / rel-L2 0.063, loss within 6e-4; per-token logprob vs stock BF16 max 0.61 / mean 0.0195 (the BF16 floor),
packed rows the same (0.69 / 0.0194). Against FP32 (`--fp32-reference` semantics, final code, 6 chains of 5-5.5k tokens): fast cosine 0.9963 / rel-L2
0.089, stock 0.9964 / 0.086, fast vs stock 0.9978 / 0.067, loss max |diff| vs FP32 0.013 / 0.009 (another 6 chains: 0.9978 / 0.066 vs 0.9973 / 0.075):
the fast path is as far from FP32 as stock HF, i.e. BF16 noise. Tried and dropped: selective recompute
of the SwiGLU product (saved activations -18%, but 1949 +- 55 vs 1987 +- 38 tok/s at equal budget).
Kernel fusion is at its limit on this hardware (layer bench, real dims, T=6000): Liger-Kernel (HF-exact `llama` RMSNorm, fused add+RMSNorm, SwiGLU;
it does not trace inside `torch.compile`) runs at the speed of inductor's kernels (non-matmul GPU ms per conv layer 12.13 vs 12.18; SwiGLU backward
2.04 vs 1.97 ms, about 234 GB/s, 85% of the GB10's 273 GB/s), so it is not used. Removing those bytes needs norm/SwiGLU fusion into the GEMMs, but
inductor's Triton GEMM templates (max-autotune, forced past its SM-count gate) run at 34-35% of cuBLAS nvjet on sm_121 and cuBLASLt has no such
epilogues; the matmul-only floor bounds what any fusion could still win at about 8% (conv layer 33.7 vs 39.5 ms, attention 43.7 vs 45.3).
BF16 matmuls at about 86-90 TFLOPs cap fwd+bwd near 9.8k tok/s with zero other work, so 10k tok/s is out of reach for 1.7B on one GB10.
GPU kernel time equals wall time per chain (no launch gaps), so CUDA graphs buy nothing at these lengths; the old per-instance layer compile already fused
(3 shared graphs, no breaks), so the gain is the sliding attention (mem-efficient masked kernel + GQA copies, about 0.47 GPU s/chain) and the CE (0.45).
Matmuls are now about 60-65% of GPU time; what remains is elementwise in the compiled layers (about 22%), flash (8%) and layer recompute.
`--stock-model` opts out. Check: `exp0_sft.py bench --fp32-reference`, `tests/neuralese/test_lfm3_fast_train.py`. The native learner
(`model/hf_port.py`) is a separate path (APPROACH_REVIEW section 1.2).

### Retired w5 machinery (removed 2026-11-12; git history keeps it)

The w5 RL controller and the native BF16 trainer stack are gone: interleaved RL with SFT windows (`rl/interleave.py`),
per-group JOLT+ teacher planning (`rl/teacher_feed.py`), Pop pre-collection of the next round (`rl/precollect.py`),
expert iteration into SFT windows with per-family window shares (`rl/rollout_distillation.py`), template withdrawal at
window boundaries (`rl/source_withdrawal.py`), memetic stages, the BF16 native SFT trainer with its transitions,
capacity proofs, recovery exports, DiLoCo and LR multiplier, CoreWeave cloud/teacher scripts, the dev16 watcher,
`behavior_trend.py` and the controller restart/rewind tools. Kept: the SGLang/native actor (`serve_native_checkpoint.py`,
`pop_rl_actor.py`), the collector, harness and scorer (`rl/task_rollouts.py`, `run_bonsai_queue.py`), `rl/held_eval.py`,
the task pool and `rl/curriculum.select_tasks`, the native server's gradient endpoints (`train/native_graph_loss.py`) and
checkpoint loading (`train/checkpoint_policy.py`). Lessons that still apply:
- A round's collector bundle was pruned after its update and w5 lost 85 successful episodes that way: keep the
  episodes an experiment trains on before pruning anything.
- Muon orthogonalizes `g + beta*m`, so its step has full size whatever the gradient's scale. w5's RL gradient was about
  1/1000 of the momentum buffer the SFT window left, so each RL round took 7-9 full-size steps along stale SFT
  momentum. Check an update's direction against its gradient before blaming the objective.
- On the GB10 a device-to-host copy into freshly allocated pageable memory runs at 0.05-0.1 GB/s (13 GB/s into
  memory the CPU already touched): prefault or pin host buffers before copying checkpoints.
- On the corrected SFT mix a backbone LR of about 2e-5 drove the model into a "repeat the failing action, ignore the
  error" attractor, and `--boundary-weight 32` put 1-9% of the objective on a few marker tokens until the model
  answered `return_result` with a native state. Never upweight a few tokens without per-token evidence.

### Task selection (engineering, 2026-11-03/04)

`rl/curriculum.select_tasks` (selector `natlang.rl-task-curriculum/2`, `exclude_ids` for the held set) starts new
families at their median level and shifts practice by outcome spread. It fills `class_quota` (effect_state 0.34) from
tagged rows: construct shares are enforced at selection, never by dropping pool candidates. It ranks over-visited rows
behind fresh ones (a ranking, never a drop). Measure the sampled mixture with `scripts/rl_tasks/simulate_selection.py`
and task-supply repetition with `scripts/rl_tasks/measure_repetition.py`.

### Task supply and meta-learning (owner, 2026-11-03)

The 2026-11-03 recipe review (`plans/neuralese/RECIPE_REVIEW_20261103/`) found that scripted templates carried about
78% of SFT loss and that RL tasks were 98% one wrapper generator. Step counts and loss on such a mixture are not
capability progress.

Owner (2026-11-03): fill measured behaviour gaps primarily with on-policy RL, not more teacher demonstrations.
Small-model (Haiku) subagents author task programs and exact oracles, prompted for real depth and diversity, and
qualified by reference/negative/alias replay. Authoring is continuous. Rotate domains and problem types. Seed designs
from existing agentic and knowledge-reasoning datasets (never protected or held inputs) and also invent original ones.
Measure repetition against the existing pool before admitting each batch. Never pad scarce behaviours by repetition.
Verified successful rollouts may self-distill into SFT.

Meta-learning is the memetic path (owner, 2026-11-03). The backbone, executing a natlang improvement program, studies
a program's invocations and failures and authors or revises its guidance, skills, prompts and adaptation choices
(including LoRA bank fits, used as a service). It is trained by on-policy RL on "improve program P" episodes, rewarded
by held-world task gains under exact oracles. This is the long-term direction. Its training returns as a controlled
addition once plain updates improve held-out success, and it is promoted by held-family task outcomes.

### Neuralese compaction is the intended runtime design (owner, 2026-11-03)

Owner intent: Neuralese compacts and summarizes large values throughout the
natlang runtime, including parameter/argument listings, large inputs, tool and
service outputs, collections and documents rendered into model context. It also
compacts context between model runs, carrying learned compressed state rather than
full text. Training data includes bgkit-derived (and other) sources transformed
into a Neuralese bottleneck: a Neuralese-prompted summarizer/extractor producer
returns Neuralese to a parent lambda/consumer that must use it for the task outcome.
The compaction and summarization operators are directly learned operators,
trained parameters rather than only prose prompts. Exact KV/prefix reuse is a
separate performance cache, not compaction. Track remaining gaps in the fallow
register until each is live in runtime, data and training.

Directly learned operators (owner, 2026-11-03) start from text: initialize each
operator (compaction, summarization/extraction, guidance, skill and other
prompt-embedding operators) from its natural-language prompt, embedded in the
real rendered call, then let it morph through direct learning on sufficient
dedicated data. Each operator needs its own adequate training data and measured
use, never a nominal parameter with a handful of rows.

Neuralese span basis (owner, 2026-11-03): every Neuralese span, soft state or
learned operator output starts from a good token sequence: input-specific,
target-relevant content the consumer needs, never a constant cue, instruction or
placeholder shared across inputs. Use gold/checked summaries or supporting spans.
Where none exist, use a deterministic mechanical summary (structure outline,
question-selected extractive spans, exact identifiers and values, length scaled to
input), recording seed provenance. Then morph through end-to-end consumer-outcome
learning with annealed text supervision. A span whose content is the same for
every input, or that ablation cannot distinguish from shuffled state, is a defect.

### Built capabilities must not lie fallow (owner, 2026-11-03)

The SGLang actor was built and integrated, then left unused for days behind a
bespoke exactness bar while RL rollouts ran about 100x too slowly on the native engine.
When the slowness surfaced, the intended solution was not raised. Standard
practice (fast-engine sampling, trainer-recomputed logprobs, truncated
importance correction, divergence rejection) was never considered. Rules:
- Keep `plans/neuralese/FALLOW_REGISTER.md` current: every built capability that
  is not in use, with its blocker, impact, next action and owner. Review it between
  experiments; its "Why these went fallow" pattern list is the checklist.
- When any symptom appears (slow, failing, poor learning), first check what we
  already built for it (register, plans, AGENTS.md) before diagnosing from scratch,
  and state that option explicitly.
- A qualification bar may not block an otherwise-ready improvement indefinitely.
  If a blocker has had no active work for 48 hours, either resolve it, replace the
  bar with the standard-practice mechanism that keeps correctness safe, or record a
  deliberate exclusion with its reason. Strictness that prevents use is not safety.
- A capability leaves the register when an experiment has measured it and it is in
  use, or when it is excluded with a reason. A commit alone is not use.

### Platform and archived findings (owner, 2026-11-09)

CoreWeave quota was exhausted on all four accounts on 2026-11-06 and is not expected back. Training runs on the DGX
(learner, SGLang actor, held eval), with Pop as a second actor and evaluator. Engineering findings from 2026-10-15 to
2026-11-02 (checkpoint/custody/capacity/numerics incidents, the Neuralese shared-state architecture, reward-policy
details, generation diversity rules) live in `plans/neuralese/ENGINEERING_FINDINGS.md`. Read the relevant section
before changing that code.

## Main goal: keep moving forward (owner, 2026-10-11)

The main goal in everything is to keep the work moving. Run the next experiment promptly and keep the machines doing
useful work for the pinned experiments (collection, training, held eval). Don't be overly cautious and don't wait for
the owner. If something is sensible, do it, record why and keep going. Raise with the owner only what is truly theirs.

- **A finding is not done until it is acted on.** A defect or a better setup becomes the change of the next
  experiment. If it is a correctness bug in a running experiment, stop that experiment and rerun it. A
  recommendation alone is an unfinished task.
- **Verify, don't assume.** Claims about state, ownership or other agents must come from a check made now:
  processes, coord activity, files, logs. Work whose author shows no activity (no process and no coord message in
  the last 24 h) is orphaned. The supervising agent owns it: review it and integrate or deliberately discard it.
  Don't address it to absent owners.
- **Diagnose surprises immediately.** An unexpected number, failure or regression gets a root-cause investigation
  in the same session, not a note for later.
- **Degradation means bug.** Don't reject an approach because our implementation of it failed (owner, 2026-10-11).
  A worse loss, degraded behaviour or an unexplained numerical gap is a bug until proven otherwise: compare against an
  exact reference, localize the divergence, fix it, and re-measure before judging the approach. When something doesn't
  make sense, run a wide, varied dragnet: data and masking, rendering and templates, dtype and upcasts, caches and
  carried state, kernels, optimizer state and resume, gradient flow, loss weighting, evaluation code, and differences
  from a known-good path. Record what was ruled out.
  Closing a degradation as "not a bug" (owner, 2026-11-14; grpo-v2-run1 step 8 was closed that way by one narrowly briefed
  subagent while the policy drifted in a consistent direction the reward did not select):
  - The explanation must be a mechanism that predicts the observed direction and size. "Noise", "learning dynamics" and
    "drift" are descriptions, not explanations; a consistent direction the objective does not select points at a biased
    gradient, i.e. a learner bug, until shown otherwise.
  - Every dragnet item above is marked checked (with evidence), ruled out, or not checked. "Found nothing in scope X" is
    recorded as exactly that, never as "no bug"; the entry stays open while any plausible item is unchecked.
  - Brief an investigator with the observations, not a hypothesis, and never with a menu whose benign answer ends the hunt
    ("bug vs learning dynamics"). Ask it to find the cause; it reports what it did not check.
  - A "not a bug" verdict is confirmed by a second, independently briefed investigator trying to find the bug (the training
    side for a training regression) before it is recorded or reported as a conclusion. Until then report it as "agent X
    found no bug in scope Y".
  - The supervisor checks a subagent's verdict against the observations before relaying it: does it explain every fact,
    above all the direction of the change?
- **Checks protect results.** Keep checks that protect task quality, numerics, held-data separation, data quality,
  recoverability and runtime function. Base-relative KL/agreement is telemetry, never a stop. Keep launch checks
  nonmutating: preflight once and let the real entry run its own checks. Never run an administrative entry in a run
  directory first (it can leave records that block the real run).
- **Code identity is provenance.** Every run, checkpoint and actor records its git commit and dirty flag
  (`common/code_provenance.py`). Refuse a wrong policy SHA, model/tokenizer identity or artifact schema/version.
  Outside an experiment, services run from the checkout `/home/werg/natlang` (Pop: `git pull`).
- **Comparable curves only.** Lower aggregate loss across changing samples, weights or quantization points is not a
  learning curve. Task outcomes on the pinned held set are.

### Work lands on main, not in worktrees (owner, 2026-10-29)

Shared-main commit hygiene: create new commits scoped to your own paths (`git commit -- PATHS`; a bare `git commit`
also takes whatever other agents staged). Never amend, reset or rebase shared main while another strand can commit;
HEAD may have changed since your last read. Reconcile divergent published history with a normal merge, preserving
both strands, rather than force-pushing.

No work may be lost or abandoned in worktrees: do work directly on main in the canonical checkout, staging only your
own paths. A worktree is a short-lived tool, not a home for work. Merge its commits to main promptly and remove it once
its commits are on main and nothing live or recorded depends on it. The pinned copy an experiment runs holds no
commits and is removed when the experiment ends.

### Agents decide; nothing waits on the owner (owner, 2026-10-29)

Operational and technical decisions such as restarting learners, restarting a
lineage from base, changing the batch or objective, or redoing work are **not**
the owner's decisions. Every agent has the autonomy to do the work the best
possible way and must not block anything on the owner. Decide, record why,
brief the other agents through coordination, and keep going. Agents still
coordinate with each other on shared resources; that is not a reason to wait
for the owner.

### On-policy collection and explanatory prompts (owner, 2026-10-29)

Collect from the actual checkpoint being updated, preserving actor/checkpoint identity and sampled token
probabilities. Historical teacher continuations and a different checkpoint's attempts are not current-policy rollouts.
RL uses the full explanatory prompt and complete tool schemas so correct idiomatic behaviour is accessible: valid
examples of awaited per-item `nl`, exact sorting/filtering/counting in code, `iterateOn` with stopping/effect checks,
direct use of inputs/services, typed concise child answers and `finish:true`. Bind the actual prompt,
tokenizer/template and runtime in both collection and training. Never put protected answers or held trajectories in
that guidance. Compact prompts are a later explicit robustness target. Measure program/template/world repetition in
what is actually trained; missing identifiers are unknown coverage, not zero repetition.

### LFM3 direction: no cross-model dependence, adaptation after a trained backbone (owner, 2026-10-28)

Strike all dependence on another model: no cross-model VAE/shared-dialect stage and no waiting for Mellum. No
additional teacher generation: use the parts of existing data that fit, and exact oracles with on-policy RL where
demonstrations are missing. Once the backbone improves, the goal is everything the tuning system needs (program tunes,
native soft states, LoRA mixtures/bases), the training corpus for the learned optimizer (metalearner), and training the
metalearner itself. Program-tuning executions and metalearner targets are collected on a backbone that has shown held
gains, never on one that has not; the machinery and oracle supply may be prepared in parallel. Order: plain
fine-tuning, RFT and GRPO until held gains, then controlled additions (Neuralese, program adaptation, metalearner).

### Lean codebase and economical testing (owner, 2026-10-28)

Always eliminate legacy infrastructure outside the core direction: code, tests,
docs, recipes and registry entries. Git history is the archive, so keep a lean
codebase with clean abstractions. Preserve source/training data, protected held
data and live recovery dependencies; never keep code "just in case".

Test economically: run the targeted tests for a change, merge, and debug problems
when they appear. Do not rerun suites on a base commit to tell pre-existing
failures from new ones. Fix every failure you see whatever its origin; when a
failing test only covers legacy infrastructure, remove the test and that code.

### Individual-program tuning (owner, 2026-10-26)

The tuning lane (`plans/TUNING_OPERATIONS_HANDOFF_20261026.md`, `plans/PROGRAM_TUNING_RUNS_20261026.md`) measures how far
each program improves through hierarchical adapters, native soft states, SFT and RL. It runs on a backbone that has
shown held gains. Continue saved adapter/soft-state optimizer histories when improving the recipe. Bind round journals
to the actual recipe, student base, initial parent and execution limits. Keep related invocation worlds on one split,
select on development and confirm on untouched confirmation worlds. Report actual base, SFT and RL task outcomes; saved
artifacts and zero-variance rollout groups are not capability lift or policy updates. Distill verified training-world
executions through ordinary source admission, preserving exact actor and split custody.

Checkpoint I/O is part of GPU supervision: track actual commits and independent exports separately from enqueue
acknowledgements, and keep cache offload or fixture preparation from starving a checkpoint write.

### Learned LoRA mixtures and bases (owner, 2026-10-27)

Make direct component-mixture and native soft-state fitting an operational part
of program adaptation. Free hierarchical LoRA fits can supply bank growth and
behavior targets, but they do not establish that mixture training or execution
has occurred. Mixture labels must come from an actual fit in the named bank,
with task outcomes measured through the composed adaptation.

Train the shared LoRA bases and component bank alongside the mixture writer,
coefficient head and native guidance in the shared learning stage. Explicitly
enable `train_basis` and `train_bank` in its operational recipe rather than
inheriting their disabled code defaults. Individual program label fits may pin a
bank snapshot; after a shared bank update, publish its actual new snapshot and
refit/rebind affected coefficient labels. Verify actual basis/component changes
and downstream behavior and preserve optimizer history. Do not describe target
extraction alone as training the learned mixture system.

### Core memetic learning and legacy removal (owner, 2026-10-23)

Memetic learning and soft skills are a core user capability and a training goal. Learn
stored native Neuralese guidance from multiple specific invocations of the same
program, measure actual task outcomes on distinct development and confirmation
invocations, and keep iterating the guidance. Keep related worlds on one side of the
split. Do not claim capability lift from stored artifacts, admission, source checks or
gradient tests alone. Its training enters as a controlled addition (see the top section).

Use the current positional KV and hybrid-state architecture in training and
inference. Stored guidance must condition the real executing model. Retain text and
tokenizer provenance for re-encoding when the executing checkpoint changes, and reuse
compatible state efficiently across repeated and concurrent calls. Guidance authoring
keeps exact actor provenance. The model being trained authors its own memetic guidance;
external teachers supply codebases and verified trajectories, not guidance (owner, 2026-10-31).

The system has no deployed-user backward-compatibility requirement. Remove legacy
adapters, compatibility shims and fallbacks, migrate their current callers and
supported functionality to the native path, and remove obsolete public entrypoints,
docs, tests and discarded-run registry entries. A legacy adapter hiding missing
functionality is work to implement, not a reason to retain the fallback. Preserve
usable source data and protected evaluation data. An explicit unsupported error is
useful during migration but is not completion of missing functionality. Keep actual
live recovery dependencies until their replacement is verified.

### Neuralese training and the LFM RL curriculum (owner, 2026-10-16, 2026-10-31)

Owner intent: actively instill Neuralese use. Mine broad task sources (with outcome oracles) for RL and
self-distillation, and prompt for Neuralese where it helps: tasks and prompt variants declaring `Neuralese<...>`
child results, per-document/per-item views, fan-in to a parent. Don't over-focus on Neuralese to the detriment of
task performance. Neuralese enters as controlled additions: Neuralese-declaring task variants first, then
whole-graph Neuralese gradients and a use bonus for correct episodes only, each measured on its own.

When Neuralese gradients are trained, the update covers the entire actual graph: preserve sampled and forced
producer turns, consumer imports, exact occurrence bindings and downstream outcomes, with gradients through the real
producer/consumer dependencies. Never turn an in-episode producer into a detached external input.

Keep an executable LFM RL task curriculum, starting with tractable verified natlang execution and growing into tool
use, multi-step work and communication. Use actual task outcomes to adapt difficulty; keep protected evaluation separate.

## Machine ownership and coordination

Owner, 2026-11-09: the coord inbox is retired. No Pop agent runs; the DGX supervisor drives Pop directly over ssh
(actor and evals). Do not check `coord.py inbox` in monitoring cycles. The coordination rules below
apply again only if a second concurrent agent is started.

- Use `/home/werg/natlang` as the canonical development checkout on DGX and Pop.
- Synchronize code through small commits pushed often to `origin/main`, using
  ordinary git: pull, merge and rebase as usual (owner, 2026-10-11), and keep the
  checkout current with main. Commit your own work promptly and stage only your
  paths; don't discard or stash away changes you didn't make.
  `scripts/git_publish.py` remains available as a helper. Do not copy source
  trees between machines. See `plans/MACHINE_COORDINATION.md`.
- Publish data additions in `training/neuralese_corpora.json` with immutable
  SHA-256 manifests under `training/corpus-manifests/`; transfer artifacts with
  `scripts/sync_training_corpora.py`. Copies, availability, and successful schema
  checks do not grant training admission. Keep split/source/quality decisions
  explicit. Register omissions and required conversions rather than losing them.
  Reconcile coverage against every entry of the exact upstream inventory, with
  explicit member/index bindings; sequential directory labels are not a coverage
  proof. Keep conversion, admission, staging and actual learner exposure separate.
- Licenses (owner, 2026-10-10): every dataset and model license is acceptable,
  including the most restrictive. Include them; record license provenance as a
  fact, but never omit or hold data for license review, and do not ask the owner.

## Storage hygiene and discarded work (standing owner order, 2026-10-11)

Owner clarification (2026-10-16): delete pretty much everything except what is
necessary to carry on. This includes remote W&B transfer/export storage. Keep
the latest independently verified recovery state, explicitly active diagnostic
or handoff parents, useful source/training data and protected held data. Prune
superseded checkpoint exports continuously after replacement durability is
verified; do not accumulate every periodic full checkpoint. Write the exact
deletion list first and verify removal. Never delete an entire mixed data and
checkpoint run merely because most of its bytes are obsolete checkpoints.

Amendment (owner, 2026-10-13): make deletions less eager. With the external disk
available for space, lean towards moving data to `/mnt/external` rather than
deleting it when unsure.

- Delete discarded approaches and runs: their checkpoints, optimizer states,
  exports, frozen runtime copies, diagnostic outputs, obsolete recipes and run
  tracking. Do not archive them, move them to another disk or machine to avoid
  deleting them, or retain them because an old note or manifest references them.
  Remove obsolete registry entries, manifests and handover tracking as well.
- Keep the active lineage and its required parents, reusable training/source
  data, and protected/held evaluation data. Distinguish these from discarded run
  artifacts before deleting; check live process references and current recipe
  dependencies. A regression still needs diagnosis before rejecting an approach.
- Deduplicate exact copies and prune superseded intermediate conversions,
  rendered packets, review previews, replays and caches. Keep one current useful
  representation with its source provenance, rather than every attempted version.
  Use existing deduplication and retention tools; preserve content hashes and
  logical paths where a current consumer needs them. Never hardlink mutable files.
- Prune rejected outputs in general. Retain one only for a concrete current use:
  an assigned repair, an unresolved bug investigation, or an actual failure/DPO
  training input. Record that use and its consumer; "might be useful someday"
  is insufficient. After the use ends, remove it. Keep independently useful
  correct rows from a partly rejected batch; remove bad rows and redundant copies.
- Check storage during the normal monitoring cadence and after large runs or
  conversions. Clean autonomously under this order. Brief the other machine so
  discarded artifacts and obsolete registry entries are removed there by its
  owner too, rather than silently surviving in mirrors. Record the current
  decision and reclaimed totals, without building another archive of discarded
  approaches. This order supersedes older blanket preservation/PRUNE-ASK rules
  for explicitly discarded work.
- These deletions are expressly desired and authorized by the owner (2026-10-11):
  deleting discarded run directories, checkpoints, exports, caches and their
  registry/manifest entries on either machine is wanted. Do it without asking for
  confirmation; list the exact paths in a script first, then delete.
## Neuralese training foundation

LFM3-1.7B is the backbone (owner, 2026-10-30; replaces the LFM2.5 QAD base and Neuralese direct v1 rules). Its port,
data curation, quantization codecs and evaluation packets are in `plans/neuralese/LFM3_RECIPE_20261028.md`. How it is
trained now is the top section of this file and `plans/neuralese/APPROACH_REVIEW_20261111.md`. A resume restores its
run's exact checkpoint and optimizer/RNG state; never substitute base weights into a resume. Qualification is actual
task behaviour, numerics and recoverable state, not similarity to the base.
