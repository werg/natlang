# Native ordinary-parameter optimizer correction

The natlang training service continues an existing model after a diagnosed
optimizer routing error. Its declared policy is: use the shared native trainer's
MuonSR latent matrices and AdamW ordinary parameters; preserve learned weights,
RNG, completed data exposure and every unaffected optimizer history. Changing
algorithms is an explicit transition, never an indistinguishable resume.

The exact mechanism extends PortMuonAdamW with a caller-declared AdamW routing
set. Native make_native_optimizer supplies all parameters returned as ordinary by
latent_split; generic legacy callers keep their existing routing. This includes
sparse vocabulary rows and narrow embedding-adapter matrices. Names and actual
serialized child slots bind all state movement.

Old Muon momentum does not contain an Adam squared-gradient history or a
per-parameter age. No exact mapping exists. The supervising owner authorized a
new Adam history for the affected parameters: zero new step/first/second moments,
while retaining the original numeric Muon buffers and group settings verbatim in
persistent transition provenance. All unaffected states and weights are exact.
No synthetic historical variance, inferred age, whole-optimizer reset or automatic
kind conversion is permitted. Default legacy loading rejects the kind mismatch;
the explicit native migration setting authorizes only the proven Muon→AdamW
ordinary-parameter change. Subsequent loads restore the new Adam history exactly.

The same explicit setting flows through native SFT/RL recipes and semantic
participant continuation. Crisp code implements identities, state durability,
exact routing and validation; learning-rate choices remain declared training
policy and are not inferred by the migration. Qualification compares independent
CE gradients, actual sparse-row/group mapping, preserved source tensors, optimizer
round trips and exact continued native updates. Production-width rare-role
negative-only and positive-target behavior is measured separately from the
migration's correctness, without loading another full student.

## Invocation and measured scope

Native SFT and native RL recipes declare
`native_optimizer_migration: native-ordinary-to-adamw-new-history-v1` (CLI:
`--native-optimizer-migration native-ordinary-to-adamw-new-history-v1`). Semantic
curricula declare the same setting globally, in global options, or per model;
phase-local declarations are rejected because participant restoration precedes
phase execution. Leaving the declaration present is idempotent after transition:
later checkpoints restore the accumulated Adam state and original provenance.
A migration-only continuation writes a checkpoint without earning data credits.

The final focused CPU ledger unit
`native-transition-final-mechanism-20261016` passed 67 tests, including both real
tiny native ports, checkpointed/in-backward versus deferred updates, exact
resumption, shared three-phase state, promotion/addition composition and a
2304-wide independent FP64 weighted-CE adjoint. The earlier private actual
M5250 selected-row diagnostic proves that inherited Muon momentum can move all
2304 coordinates even with a zero new gradient; this is not an Adam continuation
or a measured task-quality improvement. Synthetic alternating rare-positive
and ordinary-negative updates do not establish that .003 is an appropriate
Adam rate. Production rate selection and both restored-parent task measurements
remain supervising operations work, not an automatic consequence of migration.
