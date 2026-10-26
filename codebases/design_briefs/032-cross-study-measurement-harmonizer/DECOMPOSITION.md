# Cross-study measurement harmonizer

The harmonizer decides comparability before touching values. It checks construct, instrument, population, timepoint and response semantics; matching labels are insufficient. Missingness is classified separately so a valid zero is never confused with nonresponse, structural absence, censoring or unknown. Category splits require source evidence. Unit conversions require an exact receipt bound to row-collection version and hashes, especially when units change within one dataset.

Participant overlap remains uncertain absent authorized identity linkage. Only comparable measures with complete semantic and transformation evidence enter a dataset slice. Sensitivity probes report changes under supported mapping alternatives, while keeping scenarios separate from observed data. A revised mapping invalidates only derived slices that consumed its rule.


The migrated workflow contract retains the original `POL-032` typed policy record and all its values: identifier, named case-owner authority, one-case scope, 2026 effective window, empty supersession list, and owner/evidence-scope clauses. `selectHarmonizationPolicy.nl` matches that record to the requested owner, datasets and protocol before `frameHarmonization.nl` or any derived analysis proceeds. Missing or conflicting policy remains an explicit precedence blocker.
