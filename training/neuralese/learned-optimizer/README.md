# Tuning lane and joint learned optimizer on LFM3-1.7B

Owner decisions (2026-10-28, AGENTS.md "LFM3 focus"): after the LFM3 joint SFT and RL stages, the line's objective is
everything the tuning system needs (program tunes, native soft states, LoRA mixtures on the shared bases), the
learned optimizer's corpus and the optimizer itself. **No additional teacher generation**: program tunes teach the
student with its own privileged self-teaching against the programs' exact oracles and RL, and reuse existing verified
data. No cross-model dependence. Plan: [JOINT_LEARNED_OPTIMIZER.md](../../../plans/neuralese/JOINT_LEARNED_OPTIMIZER.md);
state, gaps and the compute estimate: [JOINT_LEARNED_OPTIMIZER_STATUS.md](../../../plans/neuralese/JOINT_LEARNED_OPTIMIZER_STATUS.md).

It is a lane beside the production recipe, not a stage inside its JSON: it trains no backbone weights and never
advances production lineage exposure. Its exposure is the tuned programs' adapters/soft states and the optimizer's
own parameters (writer adapter, component-token mixture head, keys, shared bank and basis).

## Run

```
training/neuralese/learned-optimizer/run-lfm3-lane.sh <trained LFM3 full checkpoint .pt> [operation id]
training/neuralese/learned-optimizer/run-lfm3-lane.sh --dry-run [operation id]   # tiny fixture, CPU, minutes
```

Start when the LFM3 joint SFT + RL checkpoint exists (any earned full checkpoint works; the operation binds its
SHA-256). The lane needs the LFM3 backbone port (native loader and server) on main. Operations live under
`LANE_ROOT` (default `/mnt/external/natlang-learned-optimizer-lfm3/operations/<id>`); every stage is its own memory
ledger unit `lopt-<stage>-<id>` with its exit status and log in the operation directory, resumable at medium
priority; rerunning an operation id skips its succeeded stages.

| Stage | What runs | Output |
| --- | --- | --- |
| `supply` | `python -m natlang_neuralese.tune.supply privileged --programs $PROGRAMS`: exact per-stage oracle references for TRAINING-partition roots of the persisted grouped splits (51 programs, 18 families) | `<program>/tune.privileged.jsonl`, `privileged-prepared.json` |
| `tune-<n>` | per program, with the lane's tuner configuration (`natlang-config/config.json`: the checkpoint as the native student, `tune.teacher: "self"`): `natlang tune P --rounds` (round 0 SFT on reused verified rows + privileged self-taught rows, then oracle-reward RL rounds with sft/rl/rft objectives, selected on development), `natlang tune P --evaluate development` (untuned vs selected release), `lane tune-result` | `tunes/<program>.json`; failures in `tunes/failed.txt` (the lane continues) |
| `specs` | family split of the tuned programs into the optimizer's corpus side and held evaluation side (25% of families); bank-init/corpus specs and train/meta templates from `lfm3-*-spec.json` | `optimizer-split.json`, specs |
| `bank-init` | shared basis (rank 16) from the corpus programs' gradients and weight SVD, one source component per program, keys from their descriptions | `bank.json` (also records the NVFP4 deploy-quantized targets) |
| `corpus` | each program's round-0 plan refit on the bank basis with update boundaries, free-tune and in-bank targets | `corpus/targets.jsonl` |
| outer cycles | `lane outer-next`: `train` up to the next bank publication (`publish_every` 100 of 400 steps, `train_bank`/`train_basis`/`train_keys` on); `adopt-bank` (the trainer continues on the published snapshot; bank moments restart with a recorded transition when basis training changed coordinates); `corpus` refit of the labels against the published bank (`cycles/<k>/corpus`) | `optimizer/state.json`, `cycles/` |
| `outer-rl` | `outer-rl-round.sh`: proposals, `natlang tune ROOT --score-proposals` per program (real task outcomes), collect, policy update | `outer-rl/` |
| `meta-examples` | exact labels for the optimizer's own stages over the final corpus plans | `meta/tune.examples.jsonl` |
| `controller-split`, `controller-supply`, `controller-tune` | persist the program-family split, prepare protected-world-checked training references, and self-teach and train the controller with SFT and oracle-reward RL | `controller/`, including its recorded active release |
| `held-learned-*`, `held-development-*`, `held-confirmation-*`, `held-baseline-*` | apply that exact controller release to held programs and measure the untuned, earlier tuned, and learned releases on development and confirmation roots | each held program's tune reports and evaluation arms |

`lane summary` writes `summary.json`: stage statuses, per-program tune outcomes (development success per round and the
untuned/active arms), the optimizer split, outer steps/publications/adoptions, meta-RL records and meta-example
counts. These are records of work done, not capability claims.

The learned spec binds the controller's program root and exact earned release. An explicit
`"policy": "crisp-baseline"` selects the non-controller baseline for comparison. Folded writes into feed-forward w1/w2/w3 are re-quantized
at the NVFP4 deploy point: measure their task effect at NVFP4 before claiming lift for the deployed model.

## Teacher-free program supply

The 51 supplied programs carry exact oracle values; the lane never generates teacher data:

1. **Reuse first.** A program's earlier verified trajectories (pool imports, earlier teacher collections) are reused
   root by root before anything new runs (`selfteach.recordedTeachers`; same validity rules as the tuner's own data).
2. **Privileged self-teaching (round 0).** On each training root without valid data, the student itself runs the
   program with the exact references as privileged context on every model turn. The recorded row is the plain task
   (the reference is never in it), admitted only when the root oracle verifies and every action is admissible; a root
   whose reply echoes the reference gives no row. This is the sequence-level form of JOLT+ privileged self-teaching.
3. **RL rounds.** Oracle-reward rollouts of the student's own release on training roots (whole-graph GRPO, RFT on
   successes, SFT anchor), selected on development roots.

Custody: references exist only for training-partition roots (`supply privileged` filters by the persisted split, the
tuner refuses a reference that names a development or confirmation root), the self teacher never runs development
roots, confirmation is measured only after the held program's learned release is selected, and family variants stay
on one side of every program split.

## Parallel placement

Program tunes are independent per program: shard the `tune-<n>` loop across one-GPU CoreWeave sandboxes (each with
the checkpoint, the programs and the lane's tuner config; results are the `tunes/*.json` files and the programs'
`.natlang/tune` directories, copied back before `specs`). Outer RL scoring (`--score-proposals`) is also per program.
bank-init, corpus cycles and outer training are sequential on one GPU (DGX or one sandbox).
