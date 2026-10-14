---
description: Construct joint worlds and consume observed evidence into a posterior.
args:
  source: BankSource
returns: Posterior
---
1. Initialize AssignmentFrontier with the original joint ticket IDs, empty worlds/rejections, and the
   actual pedigree/theory readings. The ticket inventory is finite and remains fixed.
2. Call expandAssignment.iterateOn on that state, measuring pendingTicketIds.length and finishing
   when finished is true. Consume its returned worlds and witnessed pedigree rejections.
3. Initialize one WeightedWorld per surviving ticket using its unchanged source prior mass and
   normalize over survivors. Preserve assignments and theory together in the world identity.
4. Visit actual observations in source identity order. For every currently surviving world, call
   assessObservation(world, observation, source.theories.theories, source.pedigree.inheritanceRules).
5. Call updatePosterior(posterior, assessments) with the actual decisions for that observation. Replace
   the carried posterior with its return before reading the next observation.
6. Retain assessments and rejections, including uninformative observations. An unresolved likelihood
   remains explicit and prevents asserting an exact numerical posterior for that affected branch.
7. Return the final carried Posterior. Do not collapse worlds merely because their mechanism agrees;
   a later genotype-sensitive experiment can distinguish those worlds while leaving the goal unchanged.
