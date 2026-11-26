# Failed-goal decision admission inventory

A completed internal call is execution evidence, not semantic correctness. Legacy scope/1 accepted a selected invocation when unrelated whole-packet check names were true. Its evidence digest did not establish that invocation's input/output correctness.

Consumers inventoried before edits:
- `scripts/neuralese_data/recheck_live_trajectories.py:recover`: manual scope/1 writer; completed-outcome reassessment separately verifies original failures, source contracts, dimensions and actual result.
- `scripts/neuralese_data/live_teacher_sources.py:verified_training_scope`: Python original-capture validation and per-decision eligibility. Scope/2 additionally validates exact original call/source/input/output and independent complete-result contract, refusing effectful descendants.
- `ts-host/scripts/observed-workspace-evidence.mjs:trainingScope`: independent disk/hash/source verifier.
- `ts-host/src/teacher/observed-workspace.ts:materializeObservedWorkspace`: Node IR admission and target construction after independent disk verification.
- `scripts/neuralese_data/sft_sources.py`: task-completion accounting and pool selection consume admitted episode turns; partial episodes stay task_completed=false, not completed seeds.
- `ts-host/scripts/materialize-native-teacher.mjs`: verifies observed evidence before materializing and writes no output on rejection.
- Host automatic independent-invocation-outcomes exporter emits scope/2 only; no scope/1 writer there. Whole verified goals retain independent outcome evidence, including optional-return root errors.

Bounded durable archive inventory (receipt in /home/werg/vibe-reader-eb077-rollout-ops/actual-partial-scope-inventory.json): one registered partial, GAD scope/2; zero registered scope/1. Retain original raw/context and both original failed and derivative partial statuses. No new model requests, registry registration, synthetic teacher samples or certificate fabrication.

Fix: require existing scope/2 certificates at all three admission boundaries. Manual completed reassessment remains supported; manual uncertified invocation selection fails before writing a derivative. Broader native-effect/action-only partial certification is not invented here and remains unsupported until independent per-decision evidence exists. Semantic completeness of all successful whole goals is not re-audited by this change.

Validation baseline admitted the unsupported scope/1 helper (actual ledger repro,346.7MiB). Changed-source tests:164 Python +59 Node pass; actual69 source-hashed completed seeds ->1089 eligible/0reject; actual original certified GAD partial ->1eligible,task_completed=false. Receipts and exact manifest hashes: /home/werg/vibe-certified-partial-admission-ops/actual-seeds-python-proof.json and actual-*-node-proof.json. Two optional external-file test controls are skipped in the generic Node suite; the real GAD path was explicitly supplied in the independent actual-file run.
