# Native-state SFT decomposition (2026-10-15)

## Policy

The declared `native_state_sft` stage trains the same positional-KV/hybrid-state model used by native inference. Its natural-language policy is: consume every admitted training graph and ordinary-text trajectory under its declared source/split weights; predict each supervised own-token from the actual predecessor query; retain optimizer, RNG, exposure cursor, and exact model state across operational stops and source continuations; evaluate held graphs and real tasks without making base-relative KL a gate. A stage is complete only when the declared full source exposure is complete, not when a short operational update budget ends.

## Typed functions and exact mechanisms

1. `parse_graph` (shared data function) verifies schema, tokenizer identity, split, source groups, imports, and own-token supervision. It is the trust boundary for graph rows.
2. `graph_loss` (shared model function) executes one teacher-forced graph forward and computes predecessor-query token loss. It owns exact graph masks/state placement; the trainer does not replay producer graphs.
3. `native_state_sft` (shared trainer function) owns deterministic source ordering, batching, optimizer updates, finite-gradient checks, exact resume, atomic checkpoint commits, and exposure receipts. These are crisp because optimizer/RNG persistence and token scheduling require exact state.
4. A model-family optimizer factory reuses existing optimizer and Mellum FP8/in-backward policies. The recipe declares settings; it does not branch by family in the objective.
5. Held graph loss and task cadence are evaluation functions over the same restored native checkpoint. They report task quality, held loss, numerics, runtime, and exposure. Similarity/KL from the base is telemetry only.
6. The recipe runner dispatches one typed stage and verifies the report/checkpoint identities. It remains the shared mechanism for both model recipes and continuations.
7. Role-aware conversion/admission is upstream data work: it produces native graph rows with explicit producer/consumer markers, tokenizer provenance, source groups, and protected splits. The trainer must reject ambiguous legacy flat Neuralese text rather than infer roles.

## State and continuation contract

The checkpoint contains the exact `natlang.neuralese-native-checkpoint/1` model state plus optimizer state/schema, Python/NumPy/Torch CPU and CUDA RNG state, deterministic source schedule identity and consumed prefix, update count, and report/held-evaluation state. Adding admitted sources uses an explicit continuation whose prior schedule is an exact prefix; restore optimizer and RNG without resetting progress. Every checkpoint is atomically committed before its report claims the corresponding update. A receipt exposes model/checkpoint identity, source identities and per-source actual supervised-token exposure, held-data binding, optimizer policy, and completed recipe-stage progress for later shared VAE readiness.

## Shared recipe policy

LFM and Mellum inherit one `native_state_sft` stage declaration and objective. Only backbone-inherent optimizer/memory settings may differ. Mellum uses its established MuonSR/in-backward and FP8 memory provisions; an AdamW-only 12B path is invalid. Recipe continuations append newly admitted sources while preserving all already consumed work and optimizer state. Full recipe progression and actual data exposure remain visible to the parent recipe runner.
