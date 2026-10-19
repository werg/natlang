# Geometry evidence after explicit classification migration

The policy is to retain usable whole-graph geometry evidence when an explicitly
admitted topology classification changes only row classification metadata.

1. Authenticate the original completed geometry report and full migration proof.
2. Select original migration parts for exactly the evidence source leaves.
3. Use the existing migration verifier to prove those current source entries.
4. Reconstruct the original ordered entries from authenticated prior indexes,
   retaining manifest priority, and require the original report digest.
5. Emit a successor with current entry digest and current index witnesses. Keep
   source, row counts, maximum cases and provenance; bind the original report
   and migration proof as additional provenance.
6. Let the existing geometry verifier validate the successor, including parsing
   its bounded actual maximum rows. It grants no numerical capacity itself.

These steps are exact durability/provenance verifiers and plumbing, hence crisp
Python. No model-facing instructions, parser policy, row identities, geometry,
training exposure or numerical code change.
