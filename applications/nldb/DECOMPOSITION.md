# nldb: decomposition of the pure engine, part by part

Every part of the database, with a decision:

- **fn**: its own natural-language function.
- **inline**: instructions inside its caller.
- **implicit**: left to the model.
- **crisp**: a TypeScript helper in a callable folder.
- **service** or **host**: outside the engine.

The executors are small, fast models, so rules are spelled out, and each function is one task a small model can
finish reliably.

## Status: target design

The tables below are the target decomposition. Four crisp helpers they name do not exist yet and are marked "(planned)":
`database/plan-steps.ts`, `execute/index-search.ts`, `execute/pages.ts` and `define/files.ts`. Until they do, that work
(renumbering plan steps, binary search and sorted insertion, page formatting, file moves) is done by the natural-language
functions that exist today: `database/plan.nl`, `database/execute/*.nl` and `database/define/*.nl`. The host side also has
`neuralese.ts`, which the tables below do not mention: it is the host's block storage and similarity index for `neuralese`
columns (content-addressed safetensors blocks, dialect pinning, IVF index), described in `NEURALESE.md`. Like `index.ts`
and `sqlite/` it is host code (exact storage), not a natural-language part.

## Policy

- **Natural language: a relational engine's content.**
  - understanding the request: kind, names, the SQL it means;
  - translating it to relational algebra;
  - each optimization rewrite and the estimates behind it;
  - each operator's algorithm;
  - placing rows in pages, and deciding index changes;
  - every integrity rule;
  - schema design and migration.
  Deterministic rules are natural language too when they are the database's semantics (unique keys, foreign keys,
  null handling).
- **Crisp: plumbing.**
  - reading and writing page lines in the file format;
  - binary search and sorted insertion in an index array;
  - renumbering plan steps;
  - file moves for renames.
- **SQL expression semantics are spelled out once.** That covers nulls, comparisons, LIKE, dates, arithmetic and
  functions. They go in the doc comments of the `Condition` and `Expression` types in `types.ts`. Every function that
  receives an expression sees those comments in its call's opening, and evaluates it in eval by those rules. They are
  not copied into each operator.
- **Durability is host code.** That means one writer at a time, the redo log, the atomic commit and recovery.
- **Analyses are computed by the caller and passed on.** Estimates go to the optimizer's rewrites, and resolved names
  go to SQL writing. A function can call only its own folder.

## Request handling

| Part | Decision | Unit | Why |
|---|---|---|---|
| Kind of request: question, change, schema or unclear | fn, decision | `database/classify` | A finite judgment. Its probability decides whether to ask back. |
| Threshold: below p = 0.5, ask back | inline | `database` | One rule. |
| The one clarifying question | fn | `database/clarify` | Its own task. |
| Dispatch: to schema definition or the query processor; return errors as outcomes | inline | `database` | Orchestration. |

## Query processor: from words to a plan

| Part | Decision | Unit | Why |
|---|---|---|---|
| Name resolution: each word to a table, a column, a value or a foreign-key path; ambiguity recorded as assumptions; unknown names give an error | fn | `database/resolve` | The semantic analysis. Its own data, a resolution record. |
| SQL writing from the resolved request (relative dates against today; constants in column types; conditions of meaning marked) | fn | `database/write-sql` | Models are fluent in SQL. Kept apart from resolution so that each call stays small. |
| SQL to a logical plan: a scan per table, joins by kind, filters, aggregate, HAVING, project, sort, by spelled translation rules | fn | `database/translate` | The translation is the algebra. Separate from writing SQL. |
| DML statements to `Change` records (insert rows or from a plan, update sets and target, delete target) | inline | `database/translate` | The same rules as SELECT. |

## Optimizer

| Part | Decision | Unit | Why |
|---|---|---|---|
| Cardinality estimates per step, from catalog row counts and spelled selectivities (equality on a key: 1; other equality: 10 %; range: 30 %; condition of meaning: 50 %; key join: the larger side) | fn | `database/estimate` | Its own data. Used by the next three rewrites. |
| Predicate push-down, with the rules for left, semi and anti joins; exact and semantic parts split, semantic filters placed last | fn | `database/push-down` | One rewrite. |
| Access paths: scan to lookup on an indexed column (equality before range, then fewest rows) | fn | `database/access-paths` | One rewrite. |
| Join order and methods: hash join on equality, smaller side built, most selective pair first | fn | `database/join-order` | One rewrite. |
| Drop no-op steps, renumber inputs | crisp | `database/plan-steps.ts` (planned) | Bookkeeping. |
| Sequence the rewrites | inline | `database/plan` | Orchestration. |

## Executor

| Part | Decision | Unit | Why |
|---|---|---|---|
| Run steps in order, independent steps at once; a question's rows in the statement's column order | inline | `database/execute` | Orchestration. |
| Scan: pages in order, fields renamed `alias.x`, where by the `Condition` rules | fn | `execute/scan` | An operator. |
| Lookup: find `[value, _id]` pairs for values or a range | crisp | `execute/index-search.ts` (planned) | Binary search is a low-level capability. |
| Lookup: fetch only the pages whose `_id` span covers the ids; then where | fn | `execute/lookup` | An operator. It uses the helper. |
| Filter: exact conditions first | fn | `execute/filter` | An operator. |
| Filter: conditions of meaning, judged once per distinct value | fn, decision | `execute/filter/meets` | A scored judgment, run per value in parallel. |
| Join: inner, left, semi, anti; hash or nested loop | fn | `execute/join` | An operator. |
| Aggregate: groups, count/sum/avg/min/max rules, empty input | fn | `execute/aggregate` | An operator. |
| Project: expressions by the `Expression` rules; DISTINCT | fn | `execute/project` | An operator. |
| Sort: SQLite's null order, stable; LIMIT and OFFSET | fn | `execute/sort` | An operator. |
| Row changes of an insert, update or delete (set expressions evaluated per target row) | fn | `execute/row-changes` | Its own rules. Each change sees the earlier ones. |
| Page writer: `_id` from nextId, the last page or a new one at 256 rows, update in place, delete, catalog counts | fn | `execute/write` | Storage decisions. |
| Page line format: compact JSON, `_id` first, column order | crisp | `execute/pages.ts` (planned) | Formatting. |
| Index maintenance: which pairs change per indexed column (an unchanged value keeps its pair; null is indexed) | fn | `execute/reindex` | It decides the changes. |
| Sorted insertion and removal of pairs | crisp | `execute/index-search.ts` (planned) | A low-level capability. |
| Check values: types and not-null per column | fn | `execute/check-values` | Integrity rules, spelled out. |
| Check conditions in words ("at least 0"), judged per distinct value | fn, decision | `execute/check-values/holds` | A scored judgment of a rule in words. |
| Check unique and key, through the index or the pages | fn | `execute/check-unique` | An integrity rule with its own reads. |
| Check foreign keys both ways: references exist, and deleted or changed keys are not referenced | fn | `execute/check-references` | An integrity rule with its own reads. |
| Fail the transaction with all violations | inline | `database/execute` | Orchestration. Nested `folder.apply` drops the changes. |

## Schema definition

| Part | Decision | Unit | Why |
|---|---|---|---|
| Design: tables, columns and types, keys, foreign keys, constraints, indexes, and the DDL text | fn | `define/design` | One coherent design judgment. Splitting it would lose coherence. |
| Catalog updates per change kind | inline | `database/define` | Spelled per kind. Small. |
| Rewrite rows for add, drop or rename column | fn | `define/migrate-rows` | Its own rules. Touches every page. |
| Build an index from all rows | fn | `define/build-index` | Its own task. It uses the crisp sorted helper. |
| Verify a new constraint over all stored rows | fn | `define/verify-constraint` | Integrity rules over existing data. |
| Rename or drop a table or index (file moves and deletes) | crisp | `define/files.ts` (planned) | Plumbing. |

## Host

| Part | Decision | Unit | Why |
|---|---|---|---|
| One request at a time; commit only an outcome reported done; redo log before files; recovery on open | host | `index.ts` | Exact durability. |
| SQLite engine | host | `index.ts`, `sqlite/` | The optimized variant. |
| Neuralese columns: content-addressed block storage, dialect pinning, similarity index | host | `neuralese.ts` | Exact storage and vector search over blocks (see `NEURALESE.md`). |

## Changes from today

- `parse` splits into resolve, write-sql and translate.
- `plan` splits into estimate, push-down, access-paths and join-order, plus a crisp renumbering helper.
- `execute` gains row-changes.
- `check` splits into check-values (with holds), check-unique and check-references.
- `migrate` splits into migrate-rows, build-index and verify-constraint, plus crisp file operations.
- Binary search, sorted insertion and page formatting become crisp helpers.
- The SQL expression rules move from five operators into the `Condition` and `Expression` type docs, and they are
  spelled out in full.
