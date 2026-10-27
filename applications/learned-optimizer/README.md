# Learned optimizer

The Natlang program of the joint learned optimizer (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md): `adapt.nl` revises
a program's native instruction states, skill states and LoRA mixtures together through `iterateOn`; `train.nl`
schedules outer training. Exact mechanism: `training/neuralese/natlang_neuralese/tune/learned/`.

- Apply to a program and release through the ordinary tune pipeline: `natlang tune <program> --learned spec.json`
  (crisp operator policy, the pluggable default).
- Run the natural-language episode: `natlang run applications/learned-optimizer -- adapt PLAN.json OUT --steps 4`.
- Schedule training: `natlang run applications/learned-optimizer -- train OPERATION.json --cycles 10`.
- Workers directly: `python -m natlang_neuralese.tune.learned {bank-init,targets,train,collect,propose,service,ladder} --spec SPEC`.
