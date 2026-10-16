# Recover an installed native recipe from its full saved checkpoint

`recoverTrainingPlan(fullCheckpoint, exactCurrentRecipe)` restores useful training
when the previous machine and its recipe plan are unavailable. The supervisor
chooses recovery and owns process/provider exclusivity. Recovery is not a claim
that an old plan, owner drain, current generation or stage completion exists.

1. Authenticate the full native checkpoint, its optimizer/RNG/training schema,
   saved stage, current recipe/file identities and distinct preserved lineage.
2. Resolve every input using the existing recipe resolver. Require the checkpoint's
   exact train and held source SHA bindings; later source changes use the existing
   explicit continuation/handoff mechanisms. Do not infer credits from a recipe.
3. Install a new plan on a fresh output with the actual current package inventory,
   current recipe and source bindings, plus explicit checkpoint-recovery evidence.
   Retain the saved lineage as lineage and the current recipe SHA as policy.
4. Resume the saved stage through the existing native trainer. That trainer
   restores its complete numeric optimizer, RNG, exposure and pending frames.
   Normal restart resumes the newly saved slot. No model arithmetic is changed.
5. Authenticate recovery again on initial-plan restart. An existing real resident
   generation or recipe plan is not replaced through this missing-plan route.

Crisp code is justified only for exact identity, checkpoint metadata, durability
and CLI plumbing. Existing recipe/trainer mechanisms remain the one execution
implementation. Earlier completed stages cannot be invented from a bare stage
checkpoint: this route requires the saved stage to be the first admitted stage.
