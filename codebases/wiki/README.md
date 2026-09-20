# Collaborative executable wiki

`merge_page.nl` semantically reconciles concurrent edits to stable page blocks.
The host checks transport identity, a pinned model/source/seed profile, update
coverage and source shape. Unresolved alternatives remain visible. This is a
notional CRDT experiment; it does not assert crisp convergence. Two local
replicas can run the same inputs and compare outcomes.

`run_cell.nl` runs a pinned page cell as a child source graph. Cells declare
`quickjs` or `natlang` and receive only their explicit text input. No wiki
host object or private environment is passed to the child. Results include
page and source revisions; a late result from an older page is marked stale.
Source changes clear displayed outputs. This local workbench has no network
transport or browser inference adapter yet.
