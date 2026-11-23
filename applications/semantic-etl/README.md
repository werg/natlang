# semantic-etl

Tier 1 Flagship A2: maps drifting vendor feeds into a strict target schema by synthesizing a verified crisp transform
once per drift, processing bulk rows in pure TypeScript with zero LLM calls, falling back to a leaf `.nl` only on
anomalous rows, verifying exact conservation invariants, and linking entities with `semanticMatch` + exact blocking.

```sh
natlang check applications/semantic-etl
```

## Architecture

- **Synthesize once, execute crisply (`buildMapper.nl` + `semanticRepair`)**: inspects sample `RawRow`s (fenced as
  `Untrusted<string>`), produces a `RowMappingSpec`, verifies it against schema and conservation invariants, and
  compiles it into a zero-token TypeScript transform (`>= 95%` crisp share on normal feeds).
- **Anomalous row fallback (`parseAnomalousRow.nl`)**: invoked via `semanticMap` ONLY on rows that fail the crisp
  schema validator.
- **Exact conservation verifier (`verifyConservation`)**: enforces row-count preservation, ID preservation, finite
  amounts, valid ISO currencies/dates, and monetary total conservation.
- **Entity resolution (`sameEntity.nl` + `semanticMatch`)**: exact blocking (`defaultEntityBlock`) prunes non-matching
  pairs before calibrated `readout: decision` entity matching.
