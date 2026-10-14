---
description: Apply actual observation likelihoods to joint world masses.
args:
  posterior: Posterior
  assessments: ObservationAssessment[]
returns: Posterior
---
1. Require exactly one assessment for each currently surviving world for the processed observation.
   Bind by actual worldId and retain all assessments in the output ledger.
2. For a known likelihood, multiply that world's carried mass by the returned rational value. A zero
   mass creates a witnessed WorldRejection and removes only that world from survivors.
3. For null likelihood, retain the world and record that its weight remains provisional. Preserve its
   prior carried mass for representation without claiming that provisional normalization is Bayesian evidence.
4. Sum retained exact masses and reduce fractions. If all established worlds are rejected, return an
   explicit source inconsistency; do not silently restore a favored hypothesis.
5. Normalize probability fields by total retained mass. Keep the distinction between unnormalized
   carried mass and conditional probability visible, and propagate any provisional-weight warning.
6. Return the new worlds, accumulated rejections/assessments and unresolved issues. Arithmetic applies
   the actual semantic decisions; this updater neither interprets observations nor selects theories.
