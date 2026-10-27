# Joint learned optimizer: LFM post-main-training lane

Owner decision (2026-10-27): everything for the joint learned optimizer is prepared, but no supplied-program tunes
and no optimizer runs start until **LFM main training is done**. Then this lane runs on the final LFM checkpoint.
Plan: [JOINT_LEARNED_OPTIMIZER.md](../../../plans/neuralese/JOINT_LEARNED_OPTIMIZER.md); state and gaps:
[JOINT_LEARNED_OPTIMIZER_STATUS.md](../../../plans/neuralese/JOINT_LEARNED_OPTIMIZER_STATUS.md).

It is a lane beside the production recipe, not a stage inside the production recipe JSON: the live LFM learner is
bound to its recipe identity, and this lane trains no backbone weights. Its exposure is the optimizer's own
(writer adapter, coefficient head, keys, bank); it never advances production lineage exposure.

## Gate

Start when the LFM production recipe has completed its stages and its final full checkpoint is independently
exported. Use that exact checkpoint for everything below (tune plans prepared for earlier students are rebound with
`rebase`; teacher turns are model-independent and are lowered again with the final student).

## Steps

1. **Supplied programs** (program-tuning lane owns the six-worker Luna authority): for each program in
   `/home/werg/data/program-supply-20261027/programs` (51 programs, 18 families; grouped splits already persisted,
   `splits-persisted.json`), export train/development cases against the final student (`natlang tune <program>
   --export-cases <dir>`; case lists bind the student base), run the reviewed campaign, import. Family variants
   (`name-vN`) stay on one side of every program-level split.
2. **Program tunes.** Ordinary `natlang tune` rounds per program on the final student (development selection,
   untouched confirmation). Their tune plans are the corpus input.
3. **Run this lane:** `training/neuralese/learned-optimizer/run-after-lfm-main.sh <final checkpoint .pt>`, after
   adding the imported plans of step 1 to `corpus-spec.json` `programs` in the operation directory
   `/mnt/external/natlang-learned-optimizer-20261027`. It installs the `lfm-*-spec.json` templates there and queues
   in the memory ledger, each after the previous one finished:
   - `bank-init`: shared basis (rank 16) from the programs' gradients and weight SVD, one source component per
     program, Neuralese keys from their descriptions;
   - `corpus`: each tune plan refit on the bank basis with update boundaries, then free-tune and in-bank targets
     (`natlang.joint-target/1`), resumable;
   - `train`: outer training of the optimizer on those targets;
   - `meta-examples`: exact labels for the optimizer's own stages (below).
4. **Tune the meta-learner.** Copy `applications/learned-optimizer` to a tuning project and run
   `natlang tune <copy> --examples <operation>/meta/tune.examples.jsonl` (teacher collection, SFT, RL as for any
   program). Its stages `adapt/chooseFocus`, `adapt/chooseOperator` and `adapt/shouldStop` have exact expected values:
   leave-one-binding-out hindsight focus, the counterfactually best operator measured from the same state, and
   hindsight stopping over a full-budget episode (`tune/learned/meta.py`). The examples group by program, so a
   program's episodes stay on one side of the split; programs on the optimizer's evaluation side never supply them.
   Once its tuned program has teacher rows, its plan joins the corpus like any other program: the optimizer then also
   proposes adaptations of itself. Large-scale targeted tuning of the meta-learner grows with every recorded episode.
5. **Measure.** `natlang tune <program> --learned spec.json` on development roots, the ceiling ladder (`ladder`),
   planted controls (`controls`). Capability claims need task outcomes on development and confirmation worlds.

## Resources

LFM2.5-350M FP32 on the GPU; bank-init held about 19 GB in the ledger with 5k-token reply-only rows. Every unit is
`--resumable` at medium priority, so the program-tuning lane's high-priority work preempts it; fits resume from their
saved state and labeled episodes are skipped on rerun. The operation lives on `/mnt/external` (root NVMe is nearly
full).
