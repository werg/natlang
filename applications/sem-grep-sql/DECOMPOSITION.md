# Sem-Grep-SQL (`applications/sem-grep-sql/`): Decomposition

Tier 3 flagship (WS6 §3 A8): cross-cutting semantic + relational queries over repositories, docs, and CSV/SQLite metadata tables, generalizing `applications/nldb` with content-addressed predicate caching and optional `Neuralese<T>` view hooks.

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
- Fixture rows in `fixtures/cases.json` use `split_group: "flagship:sem-grep-sql:<group>"` for held-v4 reservation custody.
