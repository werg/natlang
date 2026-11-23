# Sem-Grep-SQL (`applications/sem-grep-sql/`): Decomposition

Tier 3 flagship (WS6 §3 A8): cross-cutting semantic + relational queries over repositories, docs, and CSV/SQLite metadata tables, with content-addressed predicate caching and optional `Neuralese<T>` view hooks.

**Status.** Plan 06 names `applications/nldb` as this flagship's seed; this package does not extend it. nldb is a relational engine whose operators are natural-language functions over its own page files; here the relational part is exact TypeScript over a caller-written `QueryPlan`, and only the per-file predicate and extraction are semantic. Not built yet: turning a natural-language question into a `QueryPlan` (callers write the plan), a persistent predicate cache (the `PredicateCache` lives in memory for one process), SQLite inputs (only CSV is parsed), and a fixture pack with datasets (the earlier `fixtures/cases.json` named queries without the files and tables they run over, so it was removed).

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `query.ts`):**
  - Content hashing (`sha256Hex`, `createRepoFile`) marking every file body as `Untrusted<string>`.
  - Exact CSV parsing (`parseCsv`), hash joins (`inner` / `left`), exact filter pushdown (`eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, `includes`), ordering, limiting, and relational aggregations (`count`, `sum`, `avg`, `min`, `max` with `groupBy`).
  - Content-addressed `PredicateCache` keyed by `(sha256, predicate)` and `(sha256, question)`, deduplicating identical content within a batch and hitting ≥ 80% (100% on identical repeats) with zero redundant LLM calls.
  - Concurrency control via `semanticFilter` and `semanticMap` from `@natlang/node/std`, plus optional `Neuralese<string>` view hooks (`NeuraleseViewHook`) so native compact views can replace raw text reads per content SHA.
- **Typed semantic policy (`.nl` leaves):**
  - `matchesPredicate.nl` (`readout: decision`, returns `boolean`): judges whether an untrusted file genuinely satisfies a natural-language predicate.
  - `extractField.nl` (returns `string`): extracts a concise factual answer to a follow-up question from a matching untrusted file.

## Safety & Custody

- Every repository file body crosses the boundary as `Untrusted<string>`, fenced from prompt injection.
