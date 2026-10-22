# Restore the exact MuonSR state without duplicate transfers

The existing training recipe decides when to resume and which checkpoint and
optimizer history to restore. This change adds no policy or model-facing text.
The optimizer loader is crisp tensor transport and state binding, shared by all
backbones; it must preserve saved values, precision, groups, ownership and RNG.

1. Identify saved momentum and running-norm tensors using their existing state
   keys. Preserve the original parameter-index bindings and group metadata.
2. Let Torch restore all other state and group metadata through its existing
   loader. Transfer each protected tensor once into owned storage on the bound
   parameter's device, retaining its saved dtype.
3. If load hooks exist, preserve the existing full loader/hook ordering exactly;
   use the original restoration path rather than changing hook observations.
4. Verify the entire restored state and subsequent stochastic updates against
   the existing loader, including BF16 parameters with FP32 state, grouped
   experts, checkpoint-backed tensors, and hook behavior. Verify transfer
   elimination independently; do not claim measured production speedup without
   an actual production observation.

Crisp mechanisms: dictionary/index binding, Torch optimizer protocol, tensor
allocation and transfer, exact state/numeric verification. No optimizer routing,
learning-rate, history migration, training schedule or data eligibility changes.
