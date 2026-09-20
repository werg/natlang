# Natlang notebook

`run.nl` executes a declared cell graph through natlang's Iterate shape.
Natlang chooses among ready cells and explains bounded result samples in answer
to a question. Exact helpers limit execution to the goal's dependency closure,
recheck readiness, associate outputs with source revisions, and reject a result
from a revision newer than the run's snapshot. Whole tables and evaluator
objects stay in the native notebook host.

`NotebookWorkspace` supports explicit `sqlite` query cells and
`typescript-host` cells. SQL uses a notebook-owned in-memory SQLite database
with a read-only connection after fixture import; query rows are normalized to
portable JSON, preserving NULL separately from empty text. TypeScript cells
receive only declared dependency outputs plus a read-only query method. Their
eval context is fresh per cell, so hidden global state cannot make one cell
depend on another accidentally. Neither engine is inferred from source text.

Editing a cell increments its revision and invalidates its transitive
descendants. A dependent cell cannot run until its parents have current
outputs. The integration test joins a real SQL aggregation with a TypeScript
presentation cell, changes the SQL source, verifies invalidation and reruns,
and checks that a write-shaped SQL statement cannot mutate the database.

This is a local batch notebook. SQL cells are application-level evaluator
bindings through the TypeScript host; the natlang runtime does not yet expose
SQLite as a selected `run_code` engine. That is the concrete cross-engine
boundary to compare before promoting SQL into the engine registry. Live edit
events, background runs, persistent datasets, large table paging and a
teacher quality pilot remain product gates. Samples are truncated for the
model; output hashes and full values remain in the host.
