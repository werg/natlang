# Publication owner configuration

The declared recipe chooses publication and its role. This shared crisp verifier transports that decision into every round configuration; it does not select work, publish a checkpoint, or grant readiness.

1. Retain legacy publication selections without owner-follow metadata.
2. When any owner field is declared, require the complete role, pointer, Python runtime, and existing sealed producer selection.
3. Verify the role against the configured model, the Python runtime against the actual producer source path, and the absolute pointer and nonnegative reserve.
4. Copy exactly those verified fields into the common provider configuration. SFT, offline preference and RL use the same factory and retain those fields in stage continuations.
5. Reuse this verifier for supervised-boundary publication updates so both entry paths enforce the same contract.

This is exact configuration plumbing. The worker still independently validates its supported runtime, pointer and model before dispatching any trainer.
