# Preserve capacity evidence across the exact saved-recipe metadata repair

The owner wants useful training and prompt RL, with optimizer/RNG/exposure kept.
The actual LFM restored-optimizer MAX checks on bd542 passed, but a real RL→SFT
continuation lost its saved recipe binding. Published 8f357 repairs that metadata
seam. Its ten-line trainer hunk reads saved options, inherits an omitted binding,
and rejects an explicitly changed lineage. It changes no model, optimizer,
gradient, source parser, batching or checkpoint tensor layout.

`retainCapacityAcrossSavedRecipeRepair(measuredCapacity, reviewedRuntime)`:
1. Require the existing actual measured graph cases, unchanged hardware/backend,
   parameter layout, geometry and optimizer state.
2. Accept only the exact native_state_sft.py byte transition below; require every
   other module to match the measured source inventory.
3. Reserve at least 64 MiB for the additional host metadata and keep the existing
   memory headroom/reserve checks in both memory domains.
4. Preserve the actual measured parent and explicitly label the continued parent
   as not newly measured. Retain all transitive evidence when chaining.

Old trainer SHA: ca5acf17bbcdfd0eaeff2712e9af07a990960d8235c6ab5bb89ae2625d04321e.
New trainer SHA: 14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8.
The frozen successor is 4ea5f59a371fee5d75302f3526ee5e2544531bee. Independent
source review confirmed the hunk matches published 8f357 exactly. The actual
continuation regression passed 37 cases after the repair.

The policy above is the supervisor's explicit review. Crisp code extends the
existing continuation verifier with exact file pins; it does not infer that an
arbitrary edit to a trainer is metadata-only or qualify another GPU. Replacement
Mellum hardware still receives its own restored-state memory check.
