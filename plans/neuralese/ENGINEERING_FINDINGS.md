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

