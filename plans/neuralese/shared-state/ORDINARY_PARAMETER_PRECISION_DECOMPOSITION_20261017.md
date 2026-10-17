# Ordinary parameter update precision

The training policy is to keep every admitted backbone parameter learnable at the declared production learning rate. This audit distinguishes MuonSR matrix writes from ordinary AdamW vector/norm writes. It does not alter active cloud runtimes.

1. Inspect the shared optimizer partition and actual checkpoint parameter/moment dtypes.
2. Measure actual changed coordinates for production-sized norm vectors at the actual Mellum backbone rate, using the production PortMuonAdamW routing and an FP32 reference. Include the existing stochastic matrix writer as a control.
3. If ordinary low precision loses updates, introduce a declared small-parameter FP32 storage profile with an explicit legacy upgrade. Preserve exact parameter values, optimizer moments, RNG and exposure. Keep activation dtypes and matrix arithmetic unchanged. Check checkpoint and optimizer restore end to end before independent review.

The probe and exact migration/storage checks are crisp mechanisms: tensor dtype, numeric preservation, changed-coordinate counts and durable checkpoint identity require exact arithmetic. Recipe adoption remains a separate supervisor decision.

## Measured finding and implemented contract

The actual Mellum step12916 metadata inspection confirms input/post-attention gains (2304) and query/key gains (128) are BF16, routed to AdamW at 3e-6, with BF16 moments. A ledgered CPU production-optimizer probe found zero changed coordinates after both one and 256 updates for BF16 unit gains of either size, despite nonzero gradients. The matching FP32 reference changed every coordinate on the first update. A BF16 MuonSR matrix control changed 466/16384 coordinates on its first update. This is a concrete ordinary norm-gain storage gap, not a claim that matrix training or model capacity is broken.

`--native-ordinary-precision fp32-norms-v1` explicitly promotes recognized per-layer RMSNorm gains only. Mellum's other ordinary matrix parameters are routers already held in FP32; current LFM backbone gains are already FP32. The explicit profile works for BF16 LFM/Qwen gains too, preserving activation output dtype. Unknown profiles are rejected. Final frozen readout norms, matrix/FP8 storage and kernels remain outside this transition.

The transition preserves Parameter objects, increments their mutation version to invalidate cached parameter bindings, promotes each existing value exactly, and restores the existing named AdamW moments at equal numeric values in FP32. Optimizer step/group options, RNG, source bindings, visits, global step and carried recipe deficits remain intact. A no-update transition is saved durably. Its history is recorded once; later ordinary and semantic checkpoint loaders inherit the profile. Raw actor identity and composite checkpoint publication/restoration bind it explicitly; interface-only patches cannot disguise a backbone storage transition.

For production Mellum's 28 layers, 136192 trainable gain elements add 272384 parameter bytes and at most 544768 moment bytes (817152 bytes total). This estimate excludes temporary load/verification buffers. The change is a new numerical storage profile and requires an independently reviewed runtime and appropriate actual capacity qualification at its controlled production handoff; the earlier ordinary report-writer metadata certificate does not qualify it.

Validation uses real tiny LFM/Mellum graph losses, shared optimizer and save/load paths, a full completed-stage transition with exact RNG/credits and next-update continuation, a production-sized gain/reference migration, BF16 HF output-dtype checks, and full composite promotion/restore. No active cloud runtime is changed by this component.

Final focused qualification: ledger `native-ordinary-precision-final-cpu-20261017` passed 72 cases in 15.27s (1.1 GiB peak), including native model/raw-policy/full update/trainer regression suites. Quick ratchets (22) and TS check passed under `native-ordinary-precision-quick-20261017`. The already-upgraded checkpoint/composite case is idempotent; serialized BF16 norm tensors falsely labeled FP32 are rejected before restoration. These CPU checks do not qualify a new production FP8/CUDA capacity profile.

Recipe integration follows the same explicit mechanism: native stage parameter declarations accept `native_ordinary_precision`, reject unknown profiles, and record an unfinished-stage tuning handoff. An upgrade handoff requires a fresh restored-model `memory_qualification` request; removing/replacing a declared profile is rejected. No old capacity certificate is inherited merely because the added storage is small. The existing trainer restores once, applies the explicit profile, runs `run_prepared` with the new model-config/runtime identity, then continues useful SGD. No second parent transfer or dedicated no-update model reload is required.
