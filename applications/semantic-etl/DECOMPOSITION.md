# Semantic ETL (`applications/semantic-etl/`): Decomposition

Tier 1 Flagship A2 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3 & §3.5): drifting vendor data pipelines that
synthesize a verified crisp row transform once per schema drift, execute bulk rows in pure TypeScript with zero
per-row LLM calls, route only anomalous rows to a semantic leaf, verify exact conservation invariants, and link
entities via `semanticMatch` with exact blocking.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `etl.ts`):**
  - `compileRowTransform`: compiles a declarative `RowMappingSpec` into a deterministic pure-TypeScript row parser
    handling JSON, RFC-4180 CSV, TSV, and key-value feeds, currency symbols, US/EU decimal separators, unit scaling,
    and ISO/US/EU dates with zero LLM calls and zero `eval`.
  - `semanticRepair`: wraps `buildMapper.nl` with `checkMappingOnSamples`, repairing the synthesized spec until every
    sample row parses cleanly and conserves totals.
  - `verifyConservation`: exact verifier checking row-count preservation, input ID preservation, finite non-NaN
    amounts, valid ISO currencies/dates, and monetary total conservation.
  - `defaultEntityBlock`: exact pre-filter blocker (tax ID, domain, country, normalized token / prefix overlap) so
    `semanticMatch` only invokes `sameEntity.nl` on plausible candidate pairs.

- **Typed semantic policy (`.nl` leaves):**
  - `buildMapper.nl`: inspects sample drifting vendor rows (fenced via `RawRow.raw: Untrusted<string>`) and returns a
    typed `RowMappingSpec` for crisp bulk execution.
  - `parseAnomalousRow.nl`: invoked ONLY on anomalous rows that fail crisp schema validation.
  - `sameEntity.nl` (`readout: decision`, `returns: boolean`): calibrated entity-resolution judgment used by
    `semanticMatch`.

## Safety Boundaries

- Every vendor payload enters as `Untrusted<string>` (`RawRow.raw`), preventing prompt injection inside vendor feeds.
- No synthesized code is ever `eval`ed: `buildMapper.nl` emits a typed `RowMappingSpec` executed by audited pure TS.
- Exact conservation invariants detect row drops, ID corruption, `NaN` values, and monetary drift automatically.
