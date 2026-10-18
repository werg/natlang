# Preserve RAW training across storage saves

Named natural-language policy: **save progress without yielding useful work**.

1. Continue the current supervised recipe stage when its checkpoint is saved for durability.
2. Yield only at an explicitly declared relative optimizer-update quantum, a completed stage, or a supervisor interruption.
3. Preserve all optimizer, RNG, source, exposure and pending-stage state across a genuine handoff. A quantum is partial recipe progress.
4. Apply the same rule to RAW seed, ordinary, foundation and adopted resident configurations for both students. Keep semantic-adapter refresh flows separate.

The observed problem was Mellum's completed but partial step21,930 operation: the shared RAW plumbing forced `refresh_at_checkpoint=True` without a relative quantum, causing a save to stop useful GPU work. LFM had already needed the corresponding operational correction.

`native_raw_rounds._supervised_quantum` is the existing exact mechanism for translating declared RAW scheduling policy into trainer parameters. It disables the legacy storage-triggered yield, preserves save cadence and any explicit relative update boundary, and reinstates the existing relative-quantum settings when declared. Resident reconfiguration calls that same mechanism. No new trainer, optimizer, prompt, reward, task or communication format is introduced.

Focused checks exercise actual emitted CLI arguments for both the no-quantum and explicit-quantum cases, and the existing RAW bootstrap/resident/round composition. Ordinary CLI opt-in and semantic-adapter refresh are unchanged. This code does not claim sandbox deployment, training completion or cross-model readiness.
