# Tuning lane and joint learned optimizer on LFM3-1.7B

Plan: [JOINT_LEARNED_OPTIMIZER.md](../../../plans/neuralese/JOINT_LEARNED_OPTIMIZER.md); current state and the
restore design: [TUNING_RESTORE_20261117.md](../../../plans/neuralese/TUNING_RESTORE_20261117.md).

The recipe stages that chained this lane (`program_adaptation`, `learned_optimizer`, `bank_metalearning`:
`tune.stage`, `tune.learned.stage`, `tune.learned.bank_stage`, `train/adaptation_stages.py`) were removed on
2026-11-12. What runs today:

- **Per-program fits and the bank, training side:** `scripts/neuralese_data/program_tune.py` (`split`, `rows`, `plan`
  -> `python -m natlang_neuralese.tune.build`, `bank`); outcomes by `held_eval measure`/`compare` on each program's
  dev split, the adapted actor served with `serve_native_checkpoint.py --adapter-release`.
- **User programs:** `natlang tune P [--rounds]`, `natlang tune P --evaluate development` (the same `tune.build`).
- **Learned optimizer operation (manual steps):** `python -m natlang_neuralese.tune.learned.lane tune-result|specs|outer-next`,
  then `python -m natlang_neuralese.tune.learned <bank-init|corpus|train|adopt-bank|propose|meta-examples|...> --spec S`
  from the `lfm3-*-spec.json` templates here; `natlang tune P --learned SPEC` deploys a release (controller program
  `applications/learned-optimizer/`, or `policy: "crisp-baseline"` for a meta-learned bank's inner fit).

None of these trains backbone weights. Their exposure is the tuned programs' adapters/soft states and the optimizer's
own parameters (writer adapter, component-token mixture head, keys, shared bank and basis).

`dry_run_fixture.py build|tune` builds a tiny checkpoint and supply programs and runs the real `tune.build` fit on
them (CPU, minutes). Folded writes into feed-forward w1/w2/w3 are re-quantized at the NVFP4 deploy point: measure
their task effect at NVFP4 before claiming lift for the deployed model.

## Program supply without a teacher

The 51 supplied programs carry exact oracle values, so tunes can run without teacher generation (teacher data is
allowed where it helps; AGENTS.md "Goal: a strong model"):

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
