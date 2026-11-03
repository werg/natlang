# Natlang supervision handoff: Codex to Claude Code

Written 2026-11-03, approximately 01:23 CET, in `/home/werg/natlang`.
The owner requested this handoff, closure of sub-agent strands, and that all OS
processes be left alone for the next supervisor. No learner was restarted or
running service stopped after that request. Encoders already running completed
naturally while the handoff was being written.

## Read this first: actual current state

**Training is stopped. There is no newly earned optimizer step68.** The selected
production parent remains behaviorally selected full67/source71. The latest R38
attempt fixed the source-reader failure, passed its numerical and largest-graph
capacity checks, and fetched16 microbatches. It then failed before SGD because
the activation-scratch reservation could not fit above the disk floor.

Do not mistake a saved R36/R38 checkpoint, decreasing historical loss, a busy
GPU during capacity qualification, or a prepared continuation for new learning.

Immediate work is to finish independent recovery publication of the two
metadata-only diagnostic saves, reclaim their raw copies/views safely, and
restart the same R38 runtime/output from the intact selected parent. Its large
capacity proof is complete and should be reused through the existing exact
same-output restart path. Do not needlessly repeat that20-minute graph test.
There is also a real opportunity to make scratch sizing specific to the actual
update instead of reserving the largest whole-source measurement every time.

CoreWeave GPU access is now enabled on all four accounts. All entitlement probe
sandboxes were stopped and independently observed completed. **No cloud learner
was launched in this pass.** The old cloud run is held because it still contains
obsolete fresh-init parents/runtime/recipe controls. Replace those controls;
do not simply remove the hold and launch them.

The latest installed Pop runtime is ready, but **Pop has no running evaluator**.
Its GPU was idle at the last check. DGX also has no current learner.

## Locations and environment

Use these abbreviations below:

```text
REPO = /home/werg/natlang
W    = /home/werg/data/lfm3-rl-dgx-20261029
OPS  = W/ops
REC  = W/private-recovery-20261101
POP  = /home/werg/data/natlang-independent-recovery-20261031  (on pop-os)
```

Main code at handoff is `cf782eb2ad19d372886c705cf739eb162e278eda`; the final
handoff documentation commit will be newer. All implementation commits below
were pushed. `git status --short` was clean before creating this document.
`git worktree list --porcelain` showed only the canonical checkout on main.

Read the actual `AGENTS.md`, particularly current owner policies. Work on main;
run actual jobs from their frozen runtime, not a mixture of canonical `scripts`
and frozen Python modules. Do not use an old worktree or stale owner acknowledgement
as evidence that someone is still handling an issue.

Local project Python: `REPO/.venv-neuralese/bin/python`.
CoreWeave/W&B orchestration Python:
`/home/werg/.local/share/natlang/cw-venv/bin/python`.
Administrative code requiring `os.pidfd_open` uses `/usr/bin/python3`.
Pop's `/usr/bin/python3` is older: **it lacks `hashlib.file_digest`**. Use bounded
read/hash loops in stdlib transfer/install scripts there.

Every substantive build/test/probe/job uses `scripts/memory_ledger.py run`.
These are asynchronous user systemd units: inspect actual service state and
journal. An admission request is not a started job or a passing test. A small
budget queued with `--wait` does not wait for a training dependency if it already
fits; use an explicit live-unit waiter when that dependency is required.

Frozen launches need matching `--workdir`, `NATLANG_REPO`, and `PYTHONPATH`.
R38 does this correctly. Use explicit OMP/MKL thread settings from its launcher.

## Running work left in place

Last direct observation: 2026-11-03 01:23 CET.

| Handle | Observed state | Meaning |
|---|---|---|
| `lfm3-intact67-full-r38-1103.service` | failed, PID0 | Stopped before SGD; scratch admission failure below. |
| `encode-r36-zero-sgd-recovery-1103.service` | inactive, success | Local exact diagnostic delta and decoded-stream verification completed. |
| `encode-r38-zero-sgd-recovery-1103.service` | inactive, success | Same, for R38. Neither new delta is privately published yet. |
| `natlang-checkpoint-offload-owner-8213332571.service` | active, PID3786456 | Global Natlang checkpoint mirror owner; leave it intact until deliberately reconciling jobs. |
| `natlang-native-storage-supervision.timer` | active/waiting | Every2minutes; observer/service runs are normally short and inactive between checks. |
| Pop `install-runtime38-v2-1103.service` | inactive, success | Runtime installed; no model loaded or evaluator launched. |

The global mirror queue is `/home/werg/data/natlang-checkpoint-offload-queue`.
Its remaining pending jobs at this observation were:

```text
1793661947102059149-3563773-3563773
  W/intact67-full-r36-1102/learner/checkpoint.pt
1793637588996678404-1927307-1927307
  W/quality-parent-onpolicy-runtime19-1102/round-000000/learner/checkpoint.pt
1793664535788608074-3644261-3644261
  W/intact67-full-r38-1103/learner/checkpoint.pt
```

The external drive has about54GiB free, with a50GiB floor. These15GB mirrors
cannot currently fit and the owner retries. A mirror retry does not mean that
training is progressing. Preserve/reconcile exact queue IDs and pins, rather
than killing the shared owner or deleting arbitrary links. Diagnostic72 already
has independent private full-state reconstruction; R36/R38 currently have the
new local deltas and still need their private publication.

The historical failed units in `systemctl list-units --all` are numerous. They
are not running workers. Query the named live handles before acting.

## Production parent, diagnostics, and behavior

Selected production full state:

```text
W/ops/bf95-quality-source-zero-sgd-v2-1102/checkpoint.pt
sha256 71c3eb45594dd866d2c3d935a5577188e7e0750bf39cb2484ebbe8e4c7d9a83b
bytes  12496833447
global step67; source-adopted metadata, not new SGD
```

Its active source has401623 graphs. Preserve568 historical positive visits,
including the one quality-deferred row. Removing a row from active sampling
does not erase its historical exposure. The inactive historical row is:

```text
text:whole-trajectory:teacher-program:ed1f058ed50882235c81:decision:0014:source-sha256:5c24c12d2c8699eff5f3bfe09a667065431321ccbf5dc38e1c8f1d5a506d984f
```

Its historical count is1. Earlier trainer resume filtered it away. That bug was
fixed before R36/R38, and the exact guard was retained.

Retained base dependencies that must not be deleted casually:

```text
W/ops/zero-sgd-source-adoption-67-full-v2-1102/checkpoint.pt
  21b4ac5ad34b5cf5ddbd280a6002537c672f81b727fc014765d1be94d790f41a
  internal full source67 state, about15.4GB
W/ops/native67-ffn-candidate-v2-1102/checkpoint.pt
  bf95bec416a1bb062999baa85df3f0d692220ef88575c22099dce008877c8960
  symlink to external full state,15405316870B
original independently backed base
  0c8e75f6bb775cd108191fbd35884e5835050a726ffe112b266be851fe2ee8fc
```

The current FFN/native-execution admission and exact recovery chains retain
these dependencies. Do not delete them to gain space or weaken their checks.

Diagnostic five-JOLT-update72 is **not** the production parent:

```text
e382ce6dedbbb042d37c831c5d49021ee6eed3965eab2ee3377bfad8f151e7da
15417358994B
W/quality-parent-onpolicy-runtime19-1102/round-000000/learner/checkpoint.pt
also W/ops/diagnostic-jolt-pre-final-stop-1102/checkpoint.pt
```

Matched protected16-case evaluation under R19: full67 scored12/16; diagnostic72
scored11/16. Candidate gained case104 but lost24 and120. Case120's numerical
answer was correct but required program repair/effect was omitted. Do not infer
a proven causal objective defect from this small comparison, but do not keep
training the worse demonstrated parent merely because it earned steps.
Selection decision: `OPS/behavior-parent-reselection-2157-1102.json`.

Full67's repeated R34 evaluation completed23:42:23 on Nov2:12/16,3227.49s,
versus R19's3224.71s. This does not establish a serving throughput gain or
per-case identity. The result is
`OPS/pop-expanded-dev16-parent67-runtime34-completed-1102.json`.
Other exact completed receipts are the `...parent67-runtime19...` and
`...jolt72-runtime19...` files beside it.

The protected packet SHA is
`6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560`;
cases0,8,...120. Keep held prompts/answers/trajectories outside training.
No current eval demonstrates improvement from the attempted R36/R38 work,
because neither attempt performed SGD.

## What was fixed and landed during this pass

1. `9444fa0b1`: cloud configuration now rebinds the exact declared runtime,
   including descriptive names such as R38 and colon-separated `PYTHONPATH`.
   The old regex only matched `runtime-<hex>` and would retain DGX paths.
   It preserves unrelated diagnostic runtime names. Also fixed raw runtime
   archive suffix custody: `archive_local_recovery_runtime.py` writes `.tar`,
   not zstd; it now rejects misleading compressed output names and declares format.
   Sixteen targeted cloud/entitlement/archive tests passed, plus2 final path-list cases.
2. `980a30b63`: bounded minimal monotone-window planning for source reads.
   It preserves whole graphs, batches and delivery order, reads each source
   forward, and rejects unbounded prefetch instead of rescanning compressed files.
3. `c4488db6c`: actual root cause of focus-reader failure. Promotion applies to
   a row's **first missing visit**, but the reader had treated all future repeats
   as focused. Rows with different earned counts then shared an invalid reader.
   Reader identity now distinguishes that first pending visit.48 targeted
   diversity/cohort/focus/frontier/stream tests passed.
4. `cf782eb2a`: `pending_focus` was supported in recipe resolution and resident
   adoption, but absent from `native_supervised.SOURCE_ARGUMENTS`. Initial seed
   commands silently omitted it. It is now passed and cleared correctly when no
   longer declared.48 targeted supervised/source-resume tests passed.

Required quick checks passed for these latest changes. Do not claim a newly
green full suite: the older full suite had real failures subsequently fixed by
targeted checks. In the prior supervision work, checkpoint-lock corrections
passed187 targeted tests, and historical exposure/resume corrections passed64.

The full actual pending plan was checked, not only small fixtures:

```text
OPS/actual-first-visit-focus-window-qualification-1103.json
401623 source rows
1045719 pending batches/read windows
maximum window524650 forward tokens, one complete batch
all reader requests monotone/distinct
all original batches, graph content, order and visit custody unchanged
2048 focused rows
first16 microbatches:190416 forward tokens
```

The first attempt at minimal buffering alone exposed a huge problem: the bad
reader identity required2658624933 buffered token positions across148694
batches. This was metadata-only measurement and was never launched. There
were exactly2 source-reader crossings, both visit1 of cohort8fdffec3..., around
indices177095/177096 and177111/177112. Their promoted and repeat requests were
about148k batches apart. Fixing first-visit reader identity eliminated the
crossings entirely for this actual plan. Do not raise the buffer cap to hide it.
Evidence: `OPS/actual-focus-reader-crossings-1103.json` and the related
`actual-focus-...measurement...` files.

## R38 controls, completed qualifications, and exact current failure

```text
runtime: W/runtime-native-first-focus-v38-1103
frozen commit: cf782eb2a
closure SHA: 1c0cf0b3cc111255c9ed2507da4a0fc622cb161179e61b79ee53b8f0683b70cf
protected union: 8ed08f067c6a9715e5f48d73bf1d5ddf5d3c07501ee100c83b361c7ae7407ef7
config: OPS/intact67-full-r38-1103/config.json
recipe/plan/wrapper: same OPS directory
output: W/intact67-full-r38-1103
owner: lfm3-intact67-full-r38-1103.service
actual prior controller PID3639218; learner3644261 (both terminated)
started00:40:22, terminal failed01:10:06
```

Actual controls:100GiB fixed ledger/kernel claim, reserve6GiB, grouping2,
BF16 stochastic rounding, quantization off, accumulation16, refresh after1
actual update, full production recipe and final task gate retained. OMP/MKL1.
The2048-row focus is bound into the initial recipe and actual `--pending-focus`
command, avoiding the earlier failed resident publication path.

The focus contains1024 admitted native producer/consumer rows and1024 teacher
program rows; it reorders useful work, not deletes source obligations. The full
401623-row recipe and quotas remain. Sources/selected IDs are in
`OPS/useful-joint-focus-1102/{focus.json,selection.json,recipe-runtime-v2.json}`.

Seven very large static graphs have268k–525k total positions but only100–109
effective completion/annotation target tokens. Raw target masks misleadingly
suggest much more supervision. Full effective-target evidence is
`OPS/seven-effective-targets-1102.json`. Their total graph size is not a
single-node model-context violation: producer nodes are at most65536 tokens.
These graphs remain obligations; useful native/program behavior is prioritized.

R38's272-token numerical reference passed00:44:43. The complete524650-token,
524782-KV-slot graph passed01:04:33 in1189.50398s. Three packed capacity cases
also passed. Proofs are under the actual learner's `memory-qualifications/`,
with `memory-qualification.json` pointing to the completed qualification.
Do not alter/relabel these files to manufacture a different qualification.

R38 then fetched16 microbatches successfully. It stopped before
`native_training_update`, at `native_state_sft.py:4291`:

```text
CheckpointSpaceUnavailable:
atomic checkpoint admission needs75184168960 free bytes
(write=32234496000, safety=42949672960);
only51965767680 are free
```

Important correction to an earlier live hypothesis: this is **not proven to be
the sum of simultaneously retained accumulation microbatches**. Inspection shows
`_measured_checkpoint_scratch_peak(memory_check, oversize_proof)` takes the maximum
scratch peak across the entire source-capacity evidence. It then adds25% and
reserves an arena of that size for the next update, even though its actual
microbatches are much smaller. The current helper does not select the smallest
qualified actual-update envelope or use the already prepared packed proof for
this choice. This conservative reservation caused the stop.

Options for the next supervisor:

- Safest immediate continuation: finish the pending diagnostic compaction below,
  calculate checkpoint primary/spare plus scratch plus floors together, and
  restart the same R38 controls/output from source71. This can reuse its exact
  completed full-source proof and start useful SGD promptly.
- Improve the shared sizing path using properly bound actual-microbatch scratch
  evidence and coverage, retaining RNG/optimizer/gradient custody. Do not assume
  a smaller scalar token count covers every graph topology or simply reduce
  the floor. A new code/runtime identity can require new qualification; avoid
  accidentally spending another20minutes on an unrelated metadata change.

There is no `train.jsonl` or successful update68 from this run. Its exceptional
save is metadata-only at67. The failed `prepare-earned68-r38-window16-1103`
correctly did not create a productive continuation. An earlier version was
queued under8GiB while training still fitted beside it and therefore ran too
early; the active-owner assertion refused it. Use an actual dependency waiter,
not memory admission as a substitute for waiting on the training handle.

## Outstanding local exact encodings: next concrete action

Both encoding units completed successfully at approximately01:20–01:21.
**Do not re-encode them. They have not yet been privately published or retired.**

| Diagnostic | Raw target SHA256 | Raw bytes | Local recovery directory | Payload bytes |
|---|---|---:|---|---:|
| R36 | `1752a1533528b5a5c9bde7cf496004ad54585011c7f4d1d8e269020a185c2744` |15403132422| `OPS/r36-zero-sgd-exact-recovery-1103` |77096573|
| R38 | `8d34b0dd2cd821c4ec96f18f6b4d854956fc652d392feeecc423cee1c83461a4` |15403121350| `OPS/r38-zero-sgd-exact-recovery-1103` |77093459|

Raw slots are respectively:

```text
W/intact67-full-r36-1102/learner/checkpoint.pt
W/intact67-full-r38-1103/learner/checkpoint.pt
```

Each recovery directory has `local-recovery-capsule.json`, exact ZIP
`receipt.json`, `payload.zlib`, copied decoders, and `stream-verification.json`.
The encoder preserved every full-state byte, optimizer/RNG/exposure included;
it granted no new training credit or complete training restore qualification.
The encoder used was `OPS/prepare-r34-diagnostic-internal-base-recovery-1103.py`.
Its name is historical; its CLI takes the actual checkpoint/hash/out.

**Its actual base is internal21b4, not external BF95.** The old BF95-base encoder
was stopped before producing an artifact because external reads were too slow.
Internal21b4 has an existing independent reconstruction chain through
`REC/native67-execution-reconstruction-capsule-1102.json`.

Useful existing private-publication template:
`OPS/publish-r34-diagnostic-exact-recovery-1103.py`.
It was successfully used for R34 with base21b4. Adapt it for R36/R38:

1. Use distinct target-specific tags/status/capsule paths and accurate zero-SGD
   diagnostic scope. Do not accidentally label the known exposure-drop bug as
   affecting these two checkpoints; that defect belonged to R34.
2. Replace its hardcoded R34 learner/control paths. R36/R38 failures may lack
   final `report.json` and `native-training-receipt.json`; preserve their actual
   controller state/config, journals and failure evidence instead.
3. Bind the actual internal-base encoder operator above, not the older original
   BF95 operator which the template still lists. The actual copied decoders and
   local capsule are authoritative.
4. Retain the template's dynamic `base_target` selection, fresh private presence
   observations, complete workstation GET, and decoded full-target stream hash.
   Do not hardcode `parent['targets']['ffn-candidate']` when the base is21b4.
5. Serialize private publishers sharing the local blob table. Do not introduce
   concurrent read-modify-write races in `rl_cloud` storage metadata.
6. Verify the completed private capsule/journal with the existing
   `_verify_private_reconstruction` checks before retiring raw sources.

A proven operational retirement template is
`OPS/retire-backed-r34-diagnostic-v3-1103.py`. **It is hardcoded to R34 and must
not be run unchanged on R36/R38.** Adapt exact source/queue IDs, owner guards,
private evidence and diagnostic classification. It fences and releases only
the managed actor-source lease, records deliberately discarded diagnostic actor
views without inventing delivery, cancels the exact redundant HDD mirror job,
and removes only verified aliases of the backed diagnostic inode. It preserves
all source data, controls, production fallback and exact reconstruction bytes.
The actor-view namespace permits no additional managed consumer for this action.
Managed registrations are serialized; process-reader inspection is only a
point-in-time observation, not a global atomic guarantee.

Its earlier failures are instructive: v1 mistook its own held owner-lock file
for a view reader; v2 omitted the retirement-history parent directory after
writing a durable intent. V3 correctly resumed that fenced intent and completed.
Do not remove a fence or discard a lease merely because an earlier operator failed.

Retire only after actual uses stop. Do not replace the production parent,
erase historical exposure, or call a diagnostic save an earned update.
Retaining a reconstructible local capsule plus independently verified private
objects prevents valuable diagnostic work being lost while reclaiming raw bytes.

## Storage recovered and current constraints

At01:23, approximately67GiB free on internal and54GiB on external. These numbers
change during activation offload and checkpoint reservations. Internal floor
is40GiB; external floor50GiB. Keep all planned copies/reservations in the same
filesystem calculation. A source can be several hardlinks to one inode.

Completed reclamation:

-1418629120B: redundant raw R36/R38 runtime tars, after fresh raw/compressed hashes
  and decoded-stream equality, with compressed copies local and private.
  Compressed archives remain; the raw `.tar` paths intentionally no longer exist.
-18861940736B: rejected R34 zero-SGD raw checkpoint and unused derived view.
  All R34 diagnostic bytes remain exactly reconstructible from the local76.98MB
  delta and independently private-GET-verified chain. Its source71 production
  fallback is untouched. Receipt `OPS/r34-backed-diagnostic-retirement-1103.json`.
  Its exact obsolete HDD mirror request was removed and the shared owner resumed.

R34 private target:

```text
e1c881132f25122c00e308bafa39ccbd6fe5a862035a2e4023a72ad44a828a6e
REC/diagnostic-r34-e1c881132f25122c-exact-delta-capsule-1102.json
capsule sha7e01b885c4369a0d891a6065f1e04575ae0aa4a06dcedf69df36383cbebb37d0
OPS/r34-diagnostic-exact-recovery-v2-1103
```

Do not rerun old R35 preparations or old helpers that require that retired raw
R34 file without deliberately restoring it. Its known missing exposure entry
was a diagnosed rejected state, not productive progress.

Common caches are already small: `/tmp/pytest-of-werg` about10MB, pip8.5MB,
uv5.9MB. LFM3 HF repository directories showed only metadata-sized local usage
at the last non-following `du`; do not assume large disposable local quantized
weights exist merely because the directory names exist.

The shared digest-history compactor was previously fixed and enabled by default
in storage supervision. It shrank606543216B to337718178B in14s using bounded
SQLite, without stat-walking the source files. The observer's real budget is2GiB.
Its surviving1.14M-key JSONL index can still be costly for tiny helpers; use a
scoped hash cache where appropriate, retaining actual content/signature checks.

## Independent recovery already complete

All below are private W&B run `werg/natlang/ulcvvcya`, through the existing local
storage account. No credential belongs in a handoff, sandbox or public log.
Use `REC/run.json` with the existing CoreWeave venv/`rl_cloud` helpers. Signed
per-object capabilities are the normal sandbox mechanism. A successful PUT or
cloud GET is not a workstation GET or complete training restore.

- Selected full67/source71:
  `REC/quality-parent-71c3eb45594dd866-exact-delta-capsule-1102.json`,
  SHA1fee66a22fe7bbc58ac2afa789943bcd1c4d0ed808a4f2aa6a893420f962fffd.
- Diagnostic72:
  `REC/earned-jolt72-e382ce6dedbbb042-exact-delta-capsule-1102.json`,
  SHAdf788acdb082ecbf68fbde398e331ea3f8434ec57bf40d069b7fb0ffb9739f7e.
- Current graph manifest/payload/map closure:
  `REC/current-graph-payload-closure-capsule-1103.json` and its export journal.
  Completed220 independently workstation-GET-verified objects,5111226867B with
  controls/capsule. Actual unique graph files:213,5108629120B, recorded in
  `OPS/cloud-current-source-inventory-1103.json`.
- Actual R38 runtime/control closure:
  `REC/runtime38-actual-continuation-capsule-1103.json`,
  SHAcc2c4bdf384f114afba4e702af2282f6a8eba56fb07f84dc9e64c0d1dc23d01c,
  9431B. Includes current controls, source/FFN admission, selector provenance,
  training-only608136658B task packet and links to graph/full-state recovery.
- Actual compressed R38 runtime:
  `OPS/runtime38-first-focus.tar.zst`,113230828B,
  SHAce21936c46456faf6fc06f6a8cd37d69e0dd5aeca2a9ba1365c0bceefd69863f.
  Raw decoded tar707522560B, SHAad166530324dc64817f851a6b6069c7b0592f113b989ac89fb7e9e2c2864a9cb.
  Exact compression/member verification receipts remain beside it.

These capsules explicitly retain `training_restore_qualified:false`; complete
graph-payload recovery does not prove an all-dependency cloud training restore.
External admission evidence, full base/environment/runtime closure and actual
restore must still be reconciled. Do not advertise a complete restore rehearsal.

## Pop: ready for the next earned-checkpoint evaluation

SSH host `pop-os` was reachable. At01:20 its GPU had0% utilization/15MiB and no
new evaluator was running. Root had about11GiB free. Keep its8GiB evaluation
headroom check and avoid unbounded checkpoint copies.

R38 installation succeeded through the bounded peer unit:

```text
POP/r38-transfer-preparation-1103/runtime38-installation.json
runtime W/runtime-native-first-focus-v38-1103  (same absolute path on Pop)
closure sha1c0cf0b3cc111255c9ed2507da4a0fc622cb161179e61b79ee53b8f0683b70cf
```

The v1 installer failed before extraction on Python3.10's missing
`hashlib.file_digest`. V2 uses a bounded loop and succeeded. Actual helper:
`OPS/install-pop-runtime38-v2-1103.py`; peer file `.../install-v2.py`.
The archive has `runtime/` and `runtime-manifest.json` prefixes, unlike the old
R34 archive. Preserve that distinction when reusing installation scripts.

Use the existing qualified model-only actor transport, not a15GB full training
checkpoint transfer for evaluation. `scripts/coreweave/send_actor_transport.py`
verifies source/view receipts and readback. The default save-owner pin can
publish one pending actor view; deliver/recover/release it through real evidence
or deliberately discard a backed diagnostic. Do not fabricate a delivery.

The existing R34 evaluator driver is
`OPS/run-pop-expanded-dev16-quality-parent-runtime34-1102.py`.
It uses the image `natlang:pop-causal-conv1d-torch214-cu134-sm89-v1`, one worker,
full prompt,2048 prefill and0.25GiB prefix cache. Peer host budget5.25GiB and
GPU envelope7.2GiB were qualified. Adapt runtime, source/transport hashes, output,
container/port names and source checkpoint honestly for a newly earned state.
Do not launch a redundant unchanged-parent54-minute baseline merely because a
training-only module changed; assess the actual comparison scope. Do evaluate
new trained checkpoints and stop selecting on aggregate loss alone.

## CoreWeave: entitlement restored, execution setup still needs integration

The one-time account check completed00:15:55 CET:
`OPS/four-cw-accounts-once-1103-0014.json`.

```text
cw1 86639df9-bbe5-475e-a471-0c747572910d  verified completed
cw2 c1691433-ec48-424f-a53c-192707e9d9fe  verified completed
cw3 aa15c7b8-b3e8-42e9-a891-d751f1ea0b04  verified completed
cw4 25948be0-6dc3-48e7-af34-0390bb3ca412  verified completed
```

Do not restart an hourly entitlement-create poller. No new Natlang training
allocation was made after these probes; this handoff is not a global inventory
of other projects' provider allocations.

The old `W` sibling `/home/werg/data/lfm3-rl-cloud/run.json` is **not** a current
production launch plan. It contains old fresh-init parents and runtime9fbae...
and remains held. Main `scripts/coreweave/rl_cloud.py` now handles exact current
runtime names, but that fix alone does not update parents or dependency closure.

Other concrete issues found while inspecting that old cloud path, not yet fixed:

- `lfm3_rl_lineage.sh` requires `trained_rounds` to advance after each controller
  call. Mandatory SFT currently returns `awaiting_semantic_refresh` with
  `pending_native_checkpoint`, without incrementing RL `trained_rounds`.
  It must not interpret useful SFT as failed/no advancement.
- `rl_cloud_sandbox.stage` skips exports using only unchanged RL round counters;
  SFT-only checkpoint advancement is not safely represented by that criterion.
  Its current checkpoint selection and metadata archive coverage also need to
  handle the actual supervised pending checkpoint/learner state.
- The old lineage setup installs the SGLang fork unconditionally, even though
  current controls use a native actor. Avoid paying for irrelevant setup.
- Current DGX seed/learner commands contain the100GiB admission wrapper and DGX
  update lock. Do not copy those host-only controls blindly into a sandbox.
- Preserve the actual current whole recipe, model/native state, optimizer/RNG,
  exposure, prompt and exact admitted inputs. The current graph closure is only
  about5.1GB; full checkpoint bases can be reconstructed at cloud bandwidth from
  existing private full-base/delta chains, avoiding repeated15GB DGX uploads.

Existing components to reuse rather than invent parallel infrastructure:
`diloco.py`, `diloco_role_supervisor.py`, `diloco_role_control.py`, `bundle.py`,
`rl_cloud_sandbox.py`, the native checkpoint/recovery/continuation modules, and
the private exact ZIP decoder. The persistent role supervisor prevents the
original bootstrap child's output-collection timeout from killing replacements.
Its actual owner PID/start and child need verification, not just a Running label.

Owner policy prefers four useful GPU learners with cohosted coordination where
measured resources permit, and quick independent recovery. First make the actual
current launch correct and productive. GPU model/VRAM filters are not required;
use measured explicit CPU/RAM minima (old1-CPU starvation was rejected), exactly
one GPU, and maximum lifetime720h with explicit idle/finished shutdown. Remaining
credit/rates are opaque; the old$500 figure is an upper bound, not a verified
balance. Do not allocate idle sandboxes while local packaging is unfinished.

## Next training/RL sequence and prepared but unearned work

The current recipe obligation is joint Natlang/native Neuralese SFT, not a
Natlang-only stage with communication deferred. Preserve the full recipe;
adopt useful new sources through explicit continuations/resident publication.
Do not wait for every possible improvement before allowing useful steps.

`OPS/prepare-earned68-r38-window16-1103.py` is only a draft for the **future actual
earned68**. It requires terminal successful SFT, report/capsule matching, a real
own scored-graph anchor, exact exposure and source receipts. It is not ready
now. It proposes accumulation4 and16-update windows to amortize startup while
preserving the actual anchor. No controller for it is live. The earlier R36 and
old earned73 scripts are superseded and must not be launched against nonexistent
or diagnostically rejected parents. Audit this R38 draft before using it.

Early full-prompt JOLT+ can interleave with useful SFT once actual trained
checkpoints provide on-policy rollouts. Use full explanations/tool schemas and
whole execution graphs. Historical actor samples/selector observations are not
new-policy rollouts or updates. No current RL or teacher generation is running.

Diagnostic72's last round had20 successful attempts over4 tasks (16 student,
4 privileged teacher), zero quality spread but nonzero efficiency advantages.
The actual native objective had small centered cost terms; a quality-only
sidecar showing zero is not proof that all update advantages were zero.
Student KL terms and tiny outcome terms need diagnosis, not a claim that they
caused the held regression. `OPS/jolt72-objective-order-review-1102.json`.
Default episode materialization now mixes roles proportionally and tasks across
windows while preserving complete graphs and sealed earned prefixes.

Four prior actor71 training-world observations are sealed in
`OPS/historical-selector-observations-ready-1102.json` and bound in actual configs.
They affect curriculum selection only, never grant current-policy rollout,
likelihood, optimizer, exposure or task-qualification credits.

Program tuning, native guidance, LoRA mixtures/bases and metalearner work remain
production obligations after the backbone has meaningful demonstrated training.
Do not revive obsolete early-step750 fits or cross-model VAE/Mellum dependencies
for this LFM3 line. Check the latest owner instructions on generation; do not
automatically restart historical six-worker Luna plans or an old paid teacher.

## Sub-agents and handoff boundaries

The only sub-agent strands visible in the current team list were already
completed; interrupt/close-out calls were issued and no new work was assigned.
All their commits are ancestors of main, independently checked at handoff:

- `terra_adjoint_tensorcore_efficiency`: bounded single-tensor RoPE,
  `06cf79a15`,41 focused tests. Actual full R38 startup capacity later passed.
- `terra_capacity_continuation_reuse`: exact ZIP XOR reconstruction avoids a
  redundant range pass, `629cc0d05`,18 ZIP and10 actor-transport tests.
- `terra_native_text_write_throughput`: independently verified private exact
  reconstruction can qualify actor-pin release, `f8178da1d`,6 CPU checks.
  Productive actor delivery is still required for the normal release path.

No running sub-agent work remains, and this checkout has no stranded worktrees.
Older agent names in inherited environment context were not live team handles.

No OS service was stopped, restarted or new training operation launched after
the owner's handoff request. The next supervisor owns publication, retirement,
training restart, cloud integration, evaluation and continuing supervision.
The goal is **not complete**. This document records both useful finished work
and the remaining blockers rather than presenting prepared artifacts as model lift.
