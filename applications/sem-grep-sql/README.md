# Sem-Grep-SQL (`@natlang/app-sem-grep-sql`)

Cross-cutting semantic and relational query engine over repository files, documentation, and CSV/SQLite metadata tables.

## Architecture

- **`types.ts`**: `RepoFile` (`content: Untrusted<string>`, `sha256`, optional `view?: Neuralese<string>`), `TableRow`, `QueryPlan`, `QueryResult`.
- **`matchesPredicate.nl`** (`readout: decision`): calibrated boolean predicate over untrusted file contents.
- **`extractField.nl`**: structured fact extraction over matched files.
- **`query.ts`**: exact relational hash joins, predicate pushdown, `PredicateCache` keyed by `(sha256, predicate)`, concurrent `semanticFilter` + `semanticMap`, exact SQL-style `groupBy`/`aggregates`, and optional `Neuralese<string>` view hook support.

## Verification

```bash
node ts-host/bin/natlang.mjs check applications/sem-grep-sql
node --test ts-host/test/flagships-tier3.test.mjs
```
