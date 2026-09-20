# Natlang repository migration workbench

`migrate.nl` lets natlang inspect a source manifest and search hits, propose
exact replacements, then evaluate a candidate revision. The host preserves
each source version in memory and materializes candidates only in temporary
check directories. It never writes a migration into the original checkout.
Each patch must match exactly once, and check results are tied to a revision.

The first entry point reports one candidate. A fuller repair loop can call
`apply` on a failed candidate and `validate` repeatedly without resetting the
source identity or altering acceptance tests. Publishing a reviewable revision
to a worktree remains a separate explicit operation. The manifest and check
commands are trusted host configuration; model output cannot add files or
arbitrary commands to them.
