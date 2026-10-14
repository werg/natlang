# Mellum held full-state resume, 2026-10-14

The executing policy is this natural-language plan. Existing `cwctl`,
`bundle.py`, bootstrap, shared recipe/trainer and private exporter implement
external calls and exact durability; there is no new trainer or scheduler.

## Decomposition

1. **Determine safe migration state** is a typed natural-language supervision
   function: compare actual progress, declared end and private verified receipt;
   select the newest fully verified checkpoint, never an unverified upload.
   Crisp helpers only read identities and exact counters.
2. **Preserve acquisition evidence** is one instruction: retain pinned catalog,
   completed SHA acquisition receipts, current progress and public HF download
   metadata; explicitly mark raw bytes that must be reacquired. Crisp file
   selection/copy and existing bundle snapshot/collect verify durability.
3. **Replace the unheld runner** is a natural-language staged procedure below.
   External SDK allocation/stop, input fetch, atomic ID bookkeeping and exact
   input/code checks are crisp mechanisms. No extra GPU runs beside training.
4. **Resume and verify** uses the existing shared full-state recipe path. Model,
   optimizer, RNG and input identities are exact verifiers; model qualification
   remains the owner's separate held CE/task gate.

## Preconditions and procedure

1. Keep old `a7858948-0e1a-4c89-bb69-e2cc29cbc243` training until checkpoint,
   metadata-survival receipt and replacement specification are reviewable by
   the DGX supervisor. Do not allocate a third GPU or SDK-stop beforehand.
2. Preserve the latest fully private-SHA-verified checkpoint and its full
   recipe-plan/identity/report/evaluation metadata. Keep original immutable
   model/head/data bindings and original lineage runtime metadata; a code-only
   handoff may update only declared package hashes. Actual image/source go in
   the replacement launch receipt. Root owns recipe admission/changes.
3. Preserve public acquisition catalog, source receipts, cache metadata and
   protected benchmark-index binding on DGX with snapshot+collect SHA checks.
   Raw public files need not fit DGX disk: if old root cannot survive, reacquire
   each exact pinned file; record this explicitly and retain training_admission
   false. Never copy account credentials or scoped download capabilities into
   this public-source evidence package.
4. After the supervisor has seen exact verified state and survival receipts,
   stop/release old allocation, verify its terminal status, then allocate one
   replacement GPU. Prefer the proven Blackwell class with at least96GB VRAM,
   4CPU/64GiB host, no volumes and NGC26.09; request minimum GPU memory explicitly
   rather than accepting an unqualified80GB device. Root disk must fit pinned
   base24.3GB, heads/data, multiple47GB atomic checkpoint copies and public data.
5. Start with ordinary bash/bootstrap as namespace-init, central stdout and
   API collect hold2592000seconds. Stage current approved code through Git
   archive, pinned public base and exact private inputs; do not copy source
   trees between machines. Fetch latest full state from its owner-refreshed
   scoped GCS capability; independently verify SHA before launch.
6. Restore the SAME absolute output directory
   `/work/runs/mellum-r8-fresh-production-20261014` with exact recipe-plan and
   rolling checkpoint. Launch existing shared recipe with `--handoff` and
   frozen bindings. Do not call this a fresh lineage or substitute LoRA/base.
   Updates after the preserved snapshot are intentionally replayed from its
   saved optimizer/RNG, rather than claimed transferred.
7. Verify actual resume step, optimizer schema, identity and first updates;
   verify device capacity and actual FP8/Muon dispatch from trainer evidence.
   Restart serialized exporter against replacement ID, preserving last-verified
   state so it skips the restored object. Update owner-side sandbox-ID file
   atomically and restart any watcher that cached the old ID.
8. Reacquire public source files with the existing bounded nice19/two-worker
   job after real startup peak permits it. Keep catalogs and receipts durable.
   Preserve final full state and all artifacts BEFORE releasing collect hold.

## Known evidence

Planned warmup end is4851 updates at0.5epoch; exact timing/counters and source
survival receipts are populated by the ledgered preparation probes below.
Current task400 is unqualified: first5/16, repeat3/16 versus inherited13base.
A held resume does not repair or hide that task regression.

## Verified preparation at21:42UTC

The bounded first1MiB cloud log confirms151,875train windows,
34,130target-bearing windows and29,781documents,0.5epoch,4851totalupdates;
16held windows. At800,4051updates remain. Recent211updates took848.40seconds,
about4.02seconds/update: roughly4.5hours of update work remains. Task800 is
currently running. Task400+repeat took87.4minutes; subsequent task/probe durations
are variable, so wall-clock completion can be many hours beyond that estimate.
There is time to replace the unsafe lifecycle deliberately.

Metadata survival was verified with existing bundle snapshot+collect:
25,561files/29,073,315metadata bytes inside an8,498,801byte tar;
bundle `7cf32aaf10120ee84940d1724706b375f9feb850df5d92ebffc70612ef241365`.
DGX path `/home/werg/data/dgxops/coreweave-mellum-metadata-survival-20261014/metadata-survival.tar.gz`;
receipt `coreweave-mellum-metadata-survival-collect-receipt-20261014.json` in
dgxops. Selection is pinned catalog/progress, per-source acquisition receipts,
public HF download metadata, protected benchmark-index binding and exact
recipe/identity/evaluation metadata. No raw412GB export or credentials were
included. Original training/output hold markers were not touched.

Latest fully verified checkpoint at this probe is510,46,821,478,103bytes,
SHA `b04223599c3c6c3fbe98801127b8d1f258f47bba44c4e718ba60d0c3fefc58dc`;
680 export is in progress. Refresh the receipt immediately before selecting
migration state; never treat an in-flight export as durable. Snapshot-on-stop
is unavailable for this existing sandbox because it was not launched with the
required file_system_snapshot mount; no call claiming root persistence was made.

Installed SDK1.19 accepts GPU request `{count:1,memory_gb:96}`. Discovery call
`list_runners(healthy_only=True,include_resources=True)` rejected current auth
with `CWSandboxAuthenticationError`; this does not establish quota or replacement
capacity. Allocate only after releasing the old GPU, then verify actual device
and root disk before fetching/launching model state. If96GB constraint is
rejected, preserve exact failure and retry a documented placement/profile;
never silently call an80GB alternative qualified.

## Refreshed preparation at22:14UTC

Both existing GPUs continue training; no replacement allocation or stop was
performed. Latest independently private-SHA-verified full state is step800,
46,821,479,895bytes, SHA
`bd4591b05077043ca9eaf8aca42a62eaa96044de5ba8231ae374865dab18689f`.
The eight-file fetch bundle including completed800 task/evaluation history is
`9fcd1643a229204a796d6e85baee7db158cfa4d3f1720c26c41c734e3e83a6dc`,
46,821,686,066bytes. Its local capability manifest is private mode600 and must
be refreshed before actual transfer; signed capabilities are not committed.

The refreshed metadata archive has26,497files,29,900,695decodedbytes and
8,787,524compressedbytes, SHA
`e8e4536803c8224916e14fdcc36771c21f901baa2ea370236f1e8cf8c3a19f01`;
existing snapshot/collect bundle
`f849501a1c9ddcd5db7d1104dba5056145bd44c4668fb8f1c1e546b947de10aa`.
D36 completed, leaving16acquired and4access-blocked sources, no pending source
in this catalog. This preserves acquisition facts, not raw-file survival.
The supervisor repaired acquisition reuse to validate every raw-file size/full
SHA and known LFS SHA before accepting an acquired receipt. Use that repaired
script on replacement; never restore historical receipt-only acquired markers
as proof that missing raw bytes exist.

Existing private allocation plumbing now refuses replacement while the old
allocation is nonterminal, records a separate candidate ID, and requests96GB
minimum GPU memory. Existing release plumbing accepts the supervisor-selected
recipe and explicit handoff; its syntax check passed under the memory ledger.
These paths have not allocated or resumed a replacement. A code-only resume
recipe and the supervisor's migration instruction remain required.

Task800 is3/16 versus inherited13base, with zero infrastructure/unresolved
outcomes; the lineage remains unqualified. LFM's separate task3000 is0/16,
within its qualified410 predecessor1/noise1 and base0, though capability is weak.
Neither successful checkpoint transport nor training progress grants promotion.

The concrete code-only continuation recipe is
`training/neuralese/recipes/neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1.json`.
Its194-file package inventory was sealed at source37697f0dff96; both the
inherited fresh recipe and actual preserved cloud plan pass
`recipe_handoff_epoch_changes` with no changes outside package code. Origin
runtime metadata, all R8 inputs, optimization and evaluation controls inherit
unchanged. Checkpoint selection remains an exact private transport receipt;
refresh the latest independently verified full state immediately before handoff.

The existing handoff requires the original frozen runtime to exist and match
its recorded inventory. All194 original file hashes were independently checked
against Git revision8c8d1e06a29d261d6cb94c94b6aeb231334c7a20. Reconstruct that
package through Git archive into the original output's runtime folder before
handoff; do not copy executable source trees from the old sandbox or omit this
check. Stage current code separately, then let existing handoff replace and
record the frozen runtime. Ledgered recipe/code-only validation passed; actual
replacement restore/update remains unexecuted.

## Public raw survival before migration

The supervisor admitted the code-only recipe and requested evaluating direct
private-cloud preservation of the sixteen acquired pinned public sources before
releasing the old root. Preserve each source as one exact archive with its
acquisition receipt and catalog identity, and independently verify the private
object SHA before declaring that source durable. Keep four access-blocked
sources explicit. No raw survival result grants training admission.

**Decomposition:** source selection is this natural-language instruction;
archive formation is crisp byte-preserving durability, checking every file
against its recorded size/full SHA during the copy; transfer is the existing
scoped private GCS exporter; the independent full-object readback is an exact
verifier. Temporary archives are removed only after a verified private receipt.
Use a single bounded nice19 archive/export stream beside the existing trainers;
checkpoint exports retain their independent lifecycle. Isolate scoped capability
filenames by object identity to avoid races. No account credentials enter cloud,
no gateway bulk download and no third GPU allocation. If the mechanism fails,
retain exact acquisition evidence and report the concrete reacquisition need
before migration, prioritizing the supervisor's D02/D21/D30/D34 conversions.

D17 direct raw survival passed at22:21:28UTC: seven files,
1,510,451,739rawbytes; tar1,510,471,680bytes, SHA
`8e25eec49234a493a3f2bbd108d83d4c31963df1c79e979dcf8d3d4accf52044`.
Private GCS PUT took14.95seconds (~101MB/s); independent full-object readback
verified the same SHA in5.25seconds. The temporary archive was removed only
after verification. The remaining fifteen acquired sources entered one
sequential nice19/ionice stream; this is not a claim that they are all durable.
Per-source receipts and terminal state are kept in
`/home/werg/data/dgxops/coreweave-mellum-raw-survival-state-20261014.json`.
Checkpoint exporter capability files now use object-SHA-specific names, and raw
receipts use separate explicit paths, preventing overlap with periodic model
exports. Latest checkpoint selection remains separate and must be refreshed
before any stop. Both existing trainers continue; no replacement allocation.

## Operational task cadence at the next safe handoff

The supervisor proposes LFM task evaluation every4000updates with CE/drift
cadence1000 unchanged, and Mellum task evaluation every1200updates with CE/drift
cadence400 unchanged. This policy reduces synchronous evaluation overhead;
existing recipe operational handoff and trainer cadence implement it. Preserve
16actual tasks, one per shape;160 is the per-case turn limit. Scoring, reference,
noise, sampling and final task evaluation remain unchanged. Keep48drift-warn/end
probes and response-KL stop1.5. Do not mutate active processes or interrupt any
current evaluation. Prepare derived recipes and verify the exact resolved diff
before the supervisor adopts either variant at a durable handoff.

Cadence candidates passed ledgered resolved-plan validation. LFM
`neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4.json` changes only
`direct_sft.parameters.task_eval_every` to4000 and updates the complete194-file
package pin for e8864dff's compressed writer addition in common/jsonio.py
(7c77db8f toceea5918). Its reader path and all immutable controls are unchanged.
Mellum `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2.json`
changes only `direct_warmup.parameters.task_eval_every` to1200 from the already
admitted code-only held-resume recipe. Both pins match current complete source.
The operational handoff verifier passes for both, with exact one-parameter
diffs after excluding the single declared LFM code-file update. Detailed receipt
is `/home/werg/data/dgxops/coreweave-task-cadence-recipes-verifier-20261014.json`.

The supervisor approved these candidates after publication/verification: LFM
may advance on its existing held allocation after a current evaluation finishes
and exact rolling optimizer/RNG state is saved; report its exact parent and
resolved diff before relaunch. Mellum uses its variant at the planned safe
replacement only after raw-survival and newest-parent availability review.
R9 data continuation remains separately admitted; these variants add no data.
