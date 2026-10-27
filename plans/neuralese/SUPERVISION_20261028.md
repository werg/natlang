# Supervision log — 2026-10-28 (dgx-claude-32b6ce1e)

Continues `SESSION_HANDOVER_20261027_NIGHT.md`. Revalidate every observation.

## Taken over (2026-10-27 23:33 Berlin)

- LFM learner advancing (40,886 → 40,931 observed); latest independent export 40,808.
- Mellum r6 sandbox `7b142220…` advancing (37,998 observed), drain 05:37 / expiry 07:37.
- Invoice whole-graph job `97f6c79fc5c1c32370c68122` is saving real updates (job step 12 at
  ~7 min/step, parent step 47). Its fit uses ~30 GiB GPU plus ~23 GiB host.

## Done

- **r6 final export bug.** `final-independent-export-after-drain-20261027.py` ast-parses
  `r6/supervise-full-state.py`, which existed for r5 but not r6: the post-drain final export would
  have crashed, so the drained head would never have been exported or the box closed. Added the
  identical `select` body (extracted verbatim from the r6 readback supervisor). The waiting
  service reads it only after drain, so no restart was needed.
- **v6 operator capsule** privately READY: the CoreWeave venv is the interpreter with `wandb`.
- **Mellum r7 rollover** prepared in `/home/werg/data/mellum-cloud-continuation-20261028-r7`.
  `build-r7-from-r6.py` derives every operator from r6 (root, unit names, `r7-recovery`
  namespace) and folds in r6's two live repairs: source materialization overlapping the
  checkpoint download, and exact-subset package aliases. New `overlay-closed-r5-r6-and-launch.py`
  overlays closed r5 and then closed r6 controls. It checks the r6 drained workspace hashes and
  writes `r7-parent-independent-recovery.json` from r6's verified final head. Then
  `resume-final-r6-r7-remote.py` (r6's resume with the parent bound at run time) continues the
  same lineage in a fresh cycle.
  `run-r7-rollover.py` waits for r6's final export, closed controls, drain binding and the r7
  operator capsule. It then allocates ONE replacement (capacity retries for up to 2 h), restores,
  overlays and launches. Adoption waits for r6's own `final-recovery-closed.json`, because r6's
  close step asserts that the shared `allocation.json`/`controller-launched.json` still name r6.
  The service is `natlang-mellum-r7-rollover-r3-20261028`, started directly with a 3 GiB cap
  and registered via `memory_ledger adopt`; ledger headroom was negative, and this recovery
  waiter must not sit in the queue. Its progress goes to `r7/rollover.log`. The r7 operator
  capsule v2 is privately READY.
  Known strictness: resume refuses a drained head outside `native_full_sft`, and refuses a drain
  inside an RL update round. Handle either explicitly if it appears.
- `cd5b91119` `run-after-lfm-main.sh`: stages are gated on their own exit status per operation
  id, and a bank publication stops with `needs-refit.json`. Lane HOLD unchanged.
- `18c21392f` learned optimizer: `policy_step` now executes with the live basis (the joint
  ratio was mixing live factors with the stale basis, and the basis got no policy gradient).
  Proposals still execute the published snapshot and record the live bank parameters. A
  negative check confirmed the new test fails without the fix.
- 47-update Invoice parent private export written and queued:
  `program-tuning-20261026/export-invoice-47-parent-archive-20261028.py`. It packs to external
  storage and does not stage the 14 GB on root. Ledger unit
  `natlang-invoice-47-parent-private-export-20261028` (low priority).

## Throughput (top engineering lever, not yet measured)

LFM trains at ~730–1,370 physical tok/s. That is roughly 3 TFLOP/s for a 350M model, because
the backbone is FP32 with TF32 disabled under the process IEEE policy. Coarse TF32/BF16 were
rejected for failing parity against FP32 tolerances, not for measured training harm.
`eval.native_train_benchmark --read-only --arithmetic ieee tf32 bf16-autocast` (uncommitted
until measured) times one complete production-policy batch per arithmetic. Against IEEE it
reports gradient relative L2, cosine and the worst per-parameter error.

The first run was stopped by the ledger guard at a 6.8 GiB peak (free memory below floor):
LFM (~46 GiB) plus Invoice (~51 GiB) leave no window. Rerun it when memory allows. If it shows
a large speedup with high gradient agreement, take it through a declared arithmetic transition:
- preserve weights, optimizer, RNG and exposure;
- requalify cached/flat behavior at tolerances appropriate to the declared arithmetic;
- watch held loss and task results.

## Resource contention

Ledger accounting is full: Invoice holds ~51 GiB and LFM ~48 GiB, leaving negative obtainable
headroom with ~18 GiB really free. Everything else queues, including:
- full `check-main`;
- the Megatoken move and retention;
- other agents' builds and tests.

Invoice SFT alone is roughly 290 steps at ~7 min/step. Revisit if this starves essential work.

## Still open (from the night handover)

Learned-bank refit/adoption controller (lane on HOLD), world-runtime custody for
ALFWorld/ScienceWorld actors, full `check-main` PASS, LFM held behavior failure (missing `kind`,
repeated `return result`): needs fresh training worlds, and the teacher needs a reviewed
one-slot plan.
