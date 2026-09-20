# Stateful API workflow

`step.nl` is a Fold step. Natlang inspects the current durable order state and
chooses the next operation or recovery action. The host rejects invalid phase
transitions, writes an intent before each external effect, records a separate
idempotent remote receipt, and preserves `pending` when an acknowledgement is
lost. The same stable key is used after restart. A reconciliation event reads
the remote receipt; an unknown result cannot be treated as failure.

The fixture has inventory reserve/release, payment charge/refund, and shipping
effects with simulated definite failure, rate limit and lost acknowledgement.
Files are atomic per write and a single host instance serializes its calls.
Cross-process locking and actual service adapter contracts are needed before
using concurrent processes or real external accounts. Credentials and client
objects remain in the host, outside the natlang value stream.
