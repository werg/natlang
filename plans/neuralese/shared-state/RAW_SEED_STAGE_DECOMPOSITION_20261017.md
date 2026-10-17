# Preserve the declared supervised stage during raw seed work

The recipe names the supervised stage whose real source visits seed a raw
policy. A failed policy collection uses that stage's existing finite supervised
quantum; it does not create an unnamed CE stage or a new exposure ledger.

1. Read the configured production stage from the authenticated raw configuration.
2. Merge complete trainer settings and explicit seed objective settings as before.
3. Bind the seed invocation's stage ID to that production stage, preserving its
   source, recipe lineage, optimizer, RNG and existing per-row exposure.
4. Keep mandatory continuation's separate, explicit stage binding authoritative
   when it selects another recipe stage.

This is exact CLI plumbing. No model-facing feedback, objective, model arithmetic,
source policy, ownership gate or scheduling decision changes. Regression evidence
must parse the actual emitted seed arguments and initialize the resident source
manager using that parsed stage; checking a dictionary alone missed this defect.
