# Natlang supervision handover — 2026-10-27, 23:28 Berlin

Owner requested wrapping up this supervisor and handing ongoing operations to another
agent. Leave useful training/export/retention processes running. This is a handover,
not completion of model training. Read **AGENTS.md** first. Revalidate every live
observation below; step numbers are observations, not timeless authority.

Machine-readable observation:
`/home/werg/data/dgxops/supervision-handover-20261027-night.json`.
Earlier operating history:
`plans/neuralese/OPERATIONS_RECOVERY_AND_STORAGE_20261027.md`.

## First actions

1. `python3 scripts/coord.py inbox --ack`; use your own reader identity if working
   alongside another supervisor. Coordinate with `dgx-claude-ux` and
   `dgx-learned-optimizer`; preserve their work and separate workloads.
2. Read actual LFM progress and Mellum heartbeat/recovery receipts below. Confirm
   the exact processes, boot/start identities, timestamps and subsequent updates.
   Never resume an older published checkpoint over a newer earned child.
3. Check Invoice job `97f6c79fc5c1c32370c68122`. It has imported substantial history
   and begun writing its initial output closure, but **no new saved optimizer
   update was observed at handover**. Inspect actual `state.json`/training logs;
   a running supervisor or archive import is not SFT/RL progress.
4. Collect the bounded-bank helper's last handoff and any uncommitted work before
   starting overlapping code edits. See the open engineering section.
5. Ensure the Mellum 05:37 drain / 07:37 expiry on October 28 and independent final
   exports remain protected. Inspect the newest objects, not just old receipts.
6. Finish pending full checks and the operator capsule, then continue performance,
   data admission and whole-recipe training. Check at least hourly; the repo's
   coordination cadence is at most 50 minutes.

## Owner directions and boundaries

- Best possible models promptly; comprehensive evolving production SFT/DPO and
  whole-graph RL on both models. Adapt runs when useful, preserving actual
  weights, optimizer/RNG and per-row exposure. Stages are not completion.
- Local LFM and **one cloud Mellum learner** remain authorized. Cloud teacher and
  cloud evaluators remain paused. CoreWeave credit/rates are opaque; $500 is an
  upper bound, not a verified remaining balance. Do not allocate unbounded jobs.
- Luna teacher generation is now **one worker**, superseding six. No live Luna
  generation worker was found. Eight declared authority targets were reduced to
  one and old multi-slot campaigns marked `drain.requested`. Preserve all completed
  source/actor/split identities. Any restart needs a reviewed one-slot plan.
  Engineering helpers are separate; Luna and GPT-6.1 Terra are authorized.
- **Learned optimizer operation HOLD:** `training/neuralese/learned-optimizer/README.md`
  and `plans/neuralese/JOINT_LEARNED_OPTIMIZER_STATUS.md` say no learned-optimizer
  runs or supplied-program tuning until the LFM production recipe completes and
  its final full checkpoint is independently exported. This is broader than the
  51-program hold. An earlier supervisor interpretation restricted the hold to
  those 51 programs; that was corrected. No mixture/bank GPU run was launched here.
  Existing Invoice/Inventory/Ticket production program-tuning work predates that
  supplied-program lane; do not discard its earned adaptations.
- Owner's latest clarification: direct mixture adaptation and native guidance
  belong in the intended system; train **shared LoRA bases and component factors**
  alongside writer/head/keys. Operational `train_basis` and `train_bank` are now
  explicitly true in both the checked-in and prepared outer recipe.
- Current ordinary program tuning is **not** proof that learned component mixtures
  are being trained. It supplies adaptations/behavior; usable coefficient targets
  require actual fits in a named bank. Publish changed banks, refit/rebind targets,
  and measure composed task execution. Do not relabel old coordinates as new ones.
- No half-solutions, legacy fallback retention, held leakage, fabricated progress,
  or base-relative KL stop rules. Keep exact producer/consumer graph gradients.

## Launching and coordination

All DGX builds/tests/probes use `scripts/memory_ledger.py run`, usually
`--wait ... --detach`, with an explicit budget, workdir and environment. Inspect
actual unit completion and logs. Numerical Python:
`/home/werg/natlang/.venv-neuralese/bin/python`.
CoreWeave SDK:
`/home/werg/.local/share/natlang/cw-venv/bin/python`.
Node:
`/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node`.
Pass `PYTHONPATH=<tested checkout>/training/neuralese:<tested checkout>` and
unit-specific `NATLANG_HASH_CACHE` for tests. Use host `/usr/bin/python3` for pidfd
ownership guards. New sandboxes require one unspecified GPU; a measured RAM
minimum is allowed. Stop only exact owned units/PIDs, never patterns.

Useful monitoring:

```bash
python3 scripts/coord.py inbox --ack
python3 scripts/memory_ledger.py status
systemctl --user status <exact-unit> --no-pager
```

Some successful transient units disappear and `journalctl --user -u UNIT` then
returns nothing despite completed logs. Retrieve journal JSON by
`_SYSTEMD_USER_UNIT` or invocation ID. Do not interpret missing transient unit as
an unrun test if its actual completed journal is available.

## LFM production learner

- Unit: `natlang-lfm-priority-reuse-r7-complete-runtime-continuation-r15-20261027`.
  Learner PID observed 1629829; revalidate. Memory claim 56 GiB, actual roughly
  40–46 GiB including GPU memory. Do not shrink its measured working envelope.
- Latest observed global step **40,873**, full SFT cycle 61. Current quantum target
  41,128; this is not full-source completion.
- Progress:
  `/home/werg/data/lfm-local-recovery-20261024/rl/cycles/cycle-000061/native_full_sft/learner/progress.json`.
  Controller cursor: same root `/rl/cycles/cursor.json`.
- Frozen runtime:
  `/home/werg/data/lfm-local-recovery-20261024/source-rejoin/priority-reuse-r7-v2-20261027/runtime/repo`.
  Runtime SHA `fc00f3a3c6cf045f1b63f5d2de8bae60ff257fe97ecb78bedf0564c502777639`;
  lineage `998f8f41117d4a6d8733bf836af66e5ef65d5a381c72c85ad0a8a484aec5fc83`.
- Native float32 masters, bf16x6 projections, tf32x3 attention, layer checkpointing,
  65,536 logical-token packs, causal-prefix sharing, FP8 off. **Checkpoint-input
  offload is disabled**; do not repeat the older mistaken offload explanation.
  Background checkpoints about 600 seconds, 1,000-update SFT quantum.
- Latest observed communication foundation visits **738 / 68,030 required**, not
  ready for cross-model VAE work. Actual SFT source 1,575,737 active rows; prior
  audit had 47,368 pending. DPO 12,144 active /269 pending; recheck exact receipts.
- Latest independently exported local full state observed **40,808**:
  `current-recovery.json` at the LFM root. SHA
  `4d205f42782db3e80ca0e94bec7c622a00d26da4f2835c3dce9517ee64b28c6c`,
  3,107,793,263 bytes. Preserve useful parent until replacement actual presence
  and full readback are confirmed.
- Export unit `natlang-lfm-correction-aware-independent-export-20261027-r5`;
  monitor `natlang-lfm-current-owner-progress-monitor-20261027`;
  retention `natlang-lfm-current-r7-reviewed-ancestry-retention-r3-20261027`.

## Mellum cloud learner and recovery deadlines

- Sandbox **7b142220-d455-46af-bf1d-d90098717970**; RTX PRO6000 Blackwell 96GB GPU,
  96GiB host RAM, approximately one CPU quota. Allocated 19:37 October27;
  **drain 05:37 / expiry 07:37 Berlin October28**. Keep concurrency at one learner.
- Boot `fc1d7366-f56b-4e53-86b9-98e14cf78758`; controller3184/start ticks9971,
  learner13691. Revalidate exact IDs. Latest observed **37,974**, cycle27 SFT,
  quantum target38,356.
- Heartbeat:
  `/home/werg/data/mellum-cloud-continuation-20261027-r6/heartbeat.json`.
  Current earned child:
  `/work/resume/training/cycles/cycle-000027/native_full_sft/learner/checkpoint.pt`.
  Published parent may still say37,356; it is not the latest earned child.
- Runtime `/work/resume/priority-reuse-r7-v2-20261027/runtime/repo`, SHA
  `58e7f81f25980c16a54894bae0c459d7058797875ec7b13bbafcfb82178157b1`;
  lineage `795ffca6bc6b803e78c2a1d67580d338685beb76f1be93fc6a2c705dbeb04489`.
  R8v3 pinned-host activation-input offload; SFT1000/DPO25 quanta, checkpoint300s.
- Latest privately readback-verified full state observed **37,866**, SHA
  `e925f527557ae9116279eeb25e1c0d5a72d62f1646acf7c9d9bbcb2c9a0b1a75`,
  48,346,162,001 bytes. Receipt authority:
  `/home/werg/data/mellum-cloud-resume-20261025/current-private-recovery.json`.
- Latest complete local recovery observed **37,357**, SHA
  `4950e0d5dbe10bddf01359b4da5fcc588e44c114b49c8e553f34290dbac24955`,
  48,347,032,657 bytes. `current-local-recovery.json` is separate authority.
  A newer local download can be in flight; do not count it before completion.
- Services, all `natlang-mellum-r6-...-20261027`: `heartbeat`, `full-state-exports`,
  `lifetime-guardian`, `closed-controls-export`, `final-full-state-export`,
  `final-verified-close`, `supplemental-snapshots`. Keep actual final verification
  before close. See scripts/receipts under the r6 root above.
- Snapshot `ad6f238e-c586-45f8-abeb-48c5af859917` failed: no qualified recovery.
  Independent exports are primary. Async cloud snapshots remain disabled because
  measured conservative available61,107,740,672B was below69,497,607,876B need.
- Root storage census earlier ~6.62TB free on7.68TB; periodic exports are48GB each.
  Rolling cloud checkpoint pin retention is not yet current-qualified; do not
  apply an old-lineage retention script blindly. Monitor growth and credit.

## Invoice whole-graph program tuning

Root `/home/werg/data/program-tuning-20261026` (P below).
The entire Invoice jobs root is already symlinked to external storage; retain the
logical path. Current job:
`P/world-root-v2-programs-20261027/invoice-review/.natlang/tune/jobs/97f6c79fc5c1c32370c68122`.

- Supervisor `natlang-invoice-current-shared-graph-supervisor-20261027`, PID3876836;
  actor `natlang-invoice-current-shared-graph-actor-20261027`, PID3877435, port8098;
  training `natlang-invoice-current-shared-graph-training-20261027`, Node3940011,
  Python3940042. Revalidate; no new saved update yet observed.
- Training40GiB claim, medium-lifetime work is resumable in ledger. Actor measured
  startup host peak3.35GiB; hard limit and ledger reduced14→10GiB, adapter cache4GiB.
  Training memory grew ~27GiB during initial import/persistence; watch actual peak.
- Frozen runtime `/tmp/natlang-current-shared-graph-program-v2-20261027/repo`,
  commit8b64ab1985c02149be40829b26a7b57822788b40; client in sibling `client/ts-host`,
  manifest9611b6b3b6709a56f074652679e70e7a524fecf6f039dd1cc19b26be11d1b378.
- Exact student base39495 SHAfe6740d1cda334e83d27c65f105e1ba1ea01c4c8b08345699f5bad1dbb9e375b.
  Actor uses its independently checked inference view, not full-state optimizer
  restore. Receipt `P/fe674-actor-view-prepared-20261027.json`.
- Resume parent has **47 actual saved adapter/soft-state updates**:
  job8607dcbde4639d68032c5e21/out, resolved
  `/mnt/external/natlang-program-tuning-20261026/active-fits/invoice-review-8607dcbde4639d68032c5e21-out`.
  State SHA5c78cc85fc85d96ca7949189d5b592d57f99ac26215de1debbe1bd0b923db575.
  Exact handoff `P/invoice-current-earned-parent-for-whole-graph-cutover-20261027.json`.
  Do not use compact inference release as trainable parent: its soft states differ.
- Launch script `P/start-current-shared-graph-invoice-20261027.py`;
  operator `P/continue-current-shared-graph-invoice-20261027.mjs`;
  config `P/config-current-shared-graph-invoice-20261027/config.json`;
  status `P/invoice-whole-graph-rl-supervision-20261027.json`.
- Full389 teacher rows,3 SFT epochs,batch4,prompt_weight0; scheduleSFT,RL,SFT,RL.
  RL4 samples,temp.8,full available epoch,objectivesSFT/RL/RFT, root-quality1 RFT,
  Adam5e-5 and stateAdam2.5e-5. Whole actual producer/consumer graphs included.
  Confirmation never gradients; development chooses release. Zero-variance
  groups/artifacts are not policy updates.
- Supervisor stops only its actor on terminal success/failure. Do not restart
  these three units without inspecting their actual saved child and reports.
- Earlier measured development: Invoice39/60 vs base0/60; Inventory18/34 vs4/34.
  These are program-specific adapted results on base39495, not backbone lift;
  untouched confirmation remains unmeasured here.

## Code integrated during this supervision interval

All changes committed; another owner pushed main, and `git fetch origin` at
23:27 found main aligned with origin/main. Recheck after writing this handover.

- `63ab8c785`: actual pending packed-SFT maximum token/KV/graph-count checks,
  geometry reuse and resident append qualification, preserving weights/RNG/
  exposure. Current-main **166 tests passed including CUDA backward**. Complete
  source KV-geometry sidecar exists; no packing deployment or speed gain claimed.
- `14f5192dc` + `a241ba4da` (Terra7204c5930/0756502de): bounded mixed archive
  admission, disk staging,256MiB fit LRU, explicit scratch cleanup, full ancestry/
  gradient/collision checks before publication. Repeated persistence caches only
  verified dependency IDs, checks the whole output/store closure and reloads only
  unseen/missing artifacts. This avoids rereading13GB historical tensors every
  save; deleted descendant repair is tested. Both ports/Adam/SGD covered.
  Current-main combined archive/family/bank/trainer suite **58 passed**.
  Not deployed into the active frozen Invoice runtime. Small fixtures do not
  establish full-history speed or total process RSS. Internal `/tmp` has only
  ~19GiB free; choose measured scratch placement, not an assumed tmpfs.
- `c0cf23c4b`: actual LoRA basis-publication bug fixed. QR of learned U,V now
  transports component factors P,Q through their R matrices, preserving
  `U P Q^T V^T`. Component IDs derive after transport. Published bank parent is
  the actual old bank, not an unmaterialized intermediate bank. Four focused
  both-port tests cover effective updates, executing native logits, and mixture
  coefficient gradients, with bank learning enabled and disabled. Live training
  coordinates are not mutated at publication.
- `c1f9d3c8f`, `2d9d0d9e6`: AGENTS and prepared recipe require direct mixtures and
  shared basis/component learning; both operational train-spec copies explicitly
  enable train_basis/train_bank/train_keys. Main recipe file:
  `training/neuralese/learned-optimizer/lfm-train-spec.json`; prepared operation
  `/mnt/external/natlang-learned-optimizer-20261027/train-spec.json`.
- Earlier this evening: complete joint state/tensor archive import, exact
  derivative dependencies, stable immutable archive links with TOCTOU checks,
  task-runtime own asset paths and Node dependency custody were integrated.
  Current frozen Invoicev2 includes these; production frozen learner source is
  older. Preserve its exact publication/runtime identities when upgrading.
- Other owner just committed `74c176bcb`: native GGUF producer adapter and role
  markers. Preserve their producer-parity work; inspect their exact validation.

## Open architectural work: learned-bank continuation

Helper `/root/terra_bounded_fit_store` finished and handed off cleanly. Its final
save/resume/publication fix was independently reviewed and integrated as
**`8623e96e1`** (helper `2a5574001`, branch `terra-learned-bank-custody`, worktree
`/tmp/natlang-terra-learned-bank`). No active helper task remains.

Implemented: exact live pre-QR basis/component parameter snapshots with bank,
basis, step, shape/dtype and native identity checks; actual Adam and CPU/CUDA RNG
restore; reject an old bank optimizer lacking its parameter snapshot; materialize
and read back a complete published bank manifest; preserve its receipt in state;
stop supervised/policy updates when publication makes labels stale. Helper tests:
9 passed (both-port uninterrupted-vs-resume and publication), plus 4 stale-boundary
checks. Current-main final rerun `natlang-final-bank-resume-integration-20261027`
completed: **9 passed in4.56seconds**. Full-main check remains separate.

Still **not implemented**:

1. An executable, resumable refit/adoption controller after publication. `train`
   still needs a structured needs-refit receipt and snapshot-bound target journals.
   The prepared400-update operation will now stop at its50-update publication
   boundary rather than silently continue stale labels. That is a diagnosed gap,
   not complete adaptation scheduling.
2. Safe adoption of newly refitted labels and a new published bank while retaining
   actual live pre-QR parameter coordinates and Adam moments. Never transport
   diagonal Adam moments as though QR were an exact optimizer transformation.
   Suggested approach: separately bind original trainable coordinates and current
   published bank; compose published artifacts with differentiable live overrides;
   verify positional component ancestry/layout before remapping changed IDs.
   Test forward/gradient equivalence and multiple resumed publication boundaries
   on both ports before use.
3. Audit policy executor and proposal paths: the policy executor currently omits
   the basis leaf override, and proposal execution uses the static bank. This can
   make live trained bank behavior differ from actual sampled/executed behavior.
4. Review recipe transitions beyond train_bank/train_basis. The new flags reject
   an implicit changed optimizer algorithm route; other recipe changes also need
   honest saved-state binding.

Do not enable the live mixture lane merely because library gradients or prepared
specs pass. Its owner hold remains. The script
`training/neuralese/learned-optimizer/run-after-lfm-main.sh` also needs operational
repair: it waits on file existence/log grep rather than bound successful process
completion, can wait forever after failure, uses fixed operation names and can
consume stale completion logs. Reuse ledger/job ownership and fresh recipe/source
bindings. Prepared external copy exists; update both through explicit deployment.
The canonical README's Luna target is now one, not six.

## Performance, capability and data priorities

- Actual logical source token census:
  `R/source-rejoin/native-source-token-work-census-20261027.json`, where
  R=`/home/werg/data/lfm-local-recovery-20261024`.
  Ordinary19,328,776,386 + communication1,331,424,156 = **20,660,200,542 logical
  forward tokens**. Recent200updates wall6160s,compute5411s,physical5,730,139,
  logical11,129,123. Prefix reuse~1.94×. Naive full pass at mixed~1807logical/s
  is about four months, **not a reliable finish estimate**. Need genuine major
  speedup, not smaller claimed coverage or a short-stage endpoint.
- Complete packed geometry:
  `R/source-rejoin/packed-geometry-lfm-r7-20261027/packed-geometry-build.json`;
  sidecar `.../packed-capacity-geometry/7921be288a3e9a21df494af2357d67a1c29bdb9ad34c363f746f8dc24298a6a9.json`,
  SHAa90508d8ad6e28d193f0f763b70b7ea8cc613985a8ee64d4e20ebe948f1ca78b.
  20,927,410,223 KV slots are **not** physical forward tokens. Do not rebuild the
  full source geometry; reuse exact verified parser/source/tokenizer bindings.
- Main packed changes are ready for actual-role/runtime-bound controller setup,
  but production adoption remains undone. Frozen controller loaders can overwrite
  shared fixes; rebind memory declarations, recipe mechanisms and export dependencies
  after the complete load. Never mutate active frozen code in place.
- Profile actual current whole-graph forward/readout/backward/optimizer, repeated
  scaffolding and physical packing. `eval.native_train_benchmark` reuses the real
  update primitive; `nsys`/`ncu` installed. Do not claim a microkernel gain as whole
  training speed. Production LFM does not use input-offload currently.
- Previous coarse BF16/TF32 failed real full-model cached parity; investigate
  actual discrepancies before adopting. Preserve FP32 native state junctions and
  optimizer masters. Existing bf16x6 is only modestly faster; hardware names alone
  do not prove an optimized run.
- Latest protected LFM health step40593: **0/2 related case variants**, both
  request-budget exhaustion. Model returns intermediate `{id,text,mayCancel}`
  arrays missing required `kind`, then repeats `return result` despite type
  feedback. This is actual behavioral failure, not two broad independent cases.
  Evidence `R/held-health-probe-internal-20261027/repaired-layout-status.json` and
  associated `step-0040593-repaired-frozen-layout/.../case-*` logs. Never train on
  these held cases; generate fresh training worlds for generalized repair.
- Earlier health attempts failed infrastructure before scoring: external cold I/O,
  frozen source-review path, missing parent npm assets. Correct shared task-runtime
  own-root asset discovery is now integrated; actual old frozen R3 review alias was
  restored with matching historical hashes. These are not model failures.
- VAE readiness still needs completed corrected producer/consumer foundations on
  both actual models; one-node SFT alone does not qualify. Keep cross-model stages
  executable and adaptive; do not claim shared semantic VAE is trained.
- Node runtime dependency custody now covers actual npm files/symlinks and oldV1
  historical readers. **Unresolved world runtime custody:** world-bridge parent
  vendor resolves outside ts-host; ALFWorld/ScienceWorld Python/Java/data bytes
  are not covered by current closure. Implement explicit actual actor closures,
  not a crawl of huge unrelated parent vendor data or revision-string proof.

## Storage, recovery supplements and pending checks

At handover root~19GiB free of916GiB; external~192GiB free of3.6TiB. Inspect
actual mount and read behavior before relying on the external disk.

- Megatoken authorized move remains incomplete:
  `/home/werg/.claude/jobs/86f8d5f1/tmp` →
  `/mnt/external/megatoken-job-workspaces/86f8d5f1/tmp`,91,715,120,939 unique bytes.
  Unit `natlang-megatoken-verified-targeted-sync-workspace-move-r6-20261027`, low
  priority/resumable, can be queued. Status under
  `P/megatoken-workspace-external-relocation-20261027/status.json`.
  Preserve originals until full readback/sync/unchanged-source/inactive-reader
  checks. Counters reset per attempt; do not treat copied_bytes as total progress.
- Previously reclaimed8,699,052,032B dead pytest scratch; do not count it again.
  Old40037pin3,107,580,207B was verified moved external with original symlink.
- Current program47-update archive (~13GB) is preserved on external storage;
  **independent private export of that whole archive remains an obligation**.
  An operator capsule contains code/proofs, not all those optimizer tensors.
- v5 operator capsule is independently privately verified:35,955,568B,
  SHA50401e43729836ee71400bda542a16fef46f7f5a51861239588bdb18e70040b7.
  `N6/restart-operators-independent-recovery-v5/ready.json`, where
  N6=`/home/werg/data/mellum-cloud-continuation-20261027-r6`.
- v6 includes current frozen whole-graph numerical runtime/client,47-parent
  handoff, current operator scripts/config. It was promoted from a low-priority
  queue to `natlang-current-operators-private-capsule-v6-r2-20261027`,2GiB/high,
  `N6/export-restart-operators-v6.py`. Initial upload failed because host Python
  lacked `wandb`; the verified archive was preserved. Retry/finalization unit
  `natlang-current-operators-private-capsule-v6-finalize-20261027` uses
  `N6/finish-existing-operators-v6-export-20261027.py` and explicitly invokes the
  numerical venv uploader; that retry **also failed with missing `wandb`**.
  Neither host `/usr/bin/python3` nor `.venv-neuralese` contains the needed package.
  Next supervisor must select the existing qualified export Python environment
  (inspect the current working private exporter command, likely the CoreWeave
  venv), verify imports, then retry the existing verified archive. No package
  installation or third retry was launched before the owner paused this goal.
  There is **no v6 private READY receipt** at handover. Check actual
  `N6/restart-operators-independent-recovery-v6/ready.json` and complete private
  readback. If interrupted before READY, inspect partial source before retry:
  script deliberately rejects silently reusing an existing source directory.
- Full `scripts/check-main.sh` unit
  `natlang-current-integration-check-main-20261027`,13GiB/low, was preempted and
  queued to resume. **No complete full-main PASS claimed**. Quick main checks
  passed (source manifest,12 ratchets,TS check). Current-main focused suites:
  packed166passed; archive/family/bank/trainer58passed; basis-publication4passed.
  Ledger admission/partial dots are not test completion. Initial wrong-path test
  launches ran zero tests and were replaced; do not cite them as verification.
- User wants commits and origin sync when suitable. Main was clean/aligned with
  origin after other owner's push at23:27. This handover and any last helper fix
  may introduce new commits; check status and complete required checks before
  your own push. Never stash/drop another owner's edits or stage broad paths.

## Final continuity rule

The next agent owns supervision. Keep actual learners/exporters running, observe
progress rather than liveness alone, resolve pending bugs and ready improvements,
and only sleep when useful processes are healthy and no actionable work is being
deferred. Do not mark the overall model-training goal complete from this handover.


## Closeout — owner paused this supervisor

The owner explicitly paused the active goal during handover. This supervisor is
stopping engineering/operations now; background training and recovery units were
left running. No model-training completion is claimed. The next supervisor must
explicitly take over the hourly checks; the paused chat agent will not keep
checking in the background.

Last helper is finished; no engineering subagent task is left active. Main is
clean apart from this handover before commit. `8623e96e1` plus this document may
be ahead of origin/main. Do not claim origin sync or full test completion without
checking them; the queued full-main check has not produced a completed PASS.
