# Durable cancellation of admission waiters

Policy remains the existing memory-admission policy: an explicit cancel withdraws the current waiting request, while a later deliberate run may request the same unit again. It does not stop an admitted execution unit or change budgets, priorities, preemption, or readiness.

1. A run remembers the unit's cancellation generation on its first locked admission transaction.
2. Cancel advances that durable generation under the same ledger lock, including when the waiting entry expired or was already removed, then removes the waiting entry.
3. Each subsequent admission poll compares its remembered generation before refreshing a queue entry or admitting work. A stale caller exits without removing a newer caller's entry or claim.
4. A fresh run reads the current generation and retains the ordinary admission checks.

Generation comparison and atomic ledger publication are crisp durability mechanisms, not a new scheduling policy. The existing detached-waiter stop remains an optimization; correctness also covers ordinary foreground `--wait` clients. Already-admitted execution belongs to the existing execution lifecycle. Legacy running Python clients need explicit retirement because changing source cannot replace their loaded loop.

The concrete incident was two obsolete foreground CUDA admission clients that recreated entries after successful cancellation, blocking useful CPU metadata staging. The exact clients were retired independently; no GPU execution unit was stopped.
