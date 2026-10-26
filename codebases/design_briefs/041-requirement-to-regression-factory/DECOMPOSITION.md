# Requirement-to-regression factory

The factory first recovers behavior contracts from requirements and approved interpretations, keeping implementation advice and unresolved intent separate. It partitions inputs and state by meaningful outcome boundaries, then authors deterministic tests whose expected behavior comes from those contracts rather than the current implementation.

Before execution, it checks sandbox scope, fixture provenance, privacy, and credential boundaries. Failure analysis aligns the trace, harness, fixture, and requirement so environment faults do not masquerade as product coverage. Load failures keep their measured timing evidence and bounded retry history. Qualification challenges the suite with named seeded defects and blocks when required behavior is undetected or an assertion was removed. Changed requirements invalidate only linked assertions until owners review new oracles.

`reconcileRegressionExecution.nl` carries actual runner traces back into a versioned execution ledger. It classifies product, oracle, fixture, harness, environment and flaky outcomes separately; a stale suite cannot update coverage and successful runs do not replace mutation qualification.
