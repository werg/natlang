# Disruption-aware shipment replanner

## Affected-leg repair with scarce-capacity receipts

The dispatcher's state is a graph of actual transport legs, reservations and customer appointments. New events first reconcile carrier operation outcomes. A partial reopening changes only eligible item classes; a later carrier rejection invalidates the dependent promise even if an earlier booking was accepted. Unaffected legs and their receipts survive repair.

NatLang constructs alternative finite paths from actual offered legs and compares the complete set of customer promises. Shared capacity is tentatively allocated in the carried book and cannot be promised until a version-bound reservation receipt exists. Cancellation, delivery and ambiguous acknowledgments are distinct lifecycle states. The customer projection preserves every dependent appointment and names the concrete promise requiring renegotiation.

## Verification boundary

The original domain records, five semantic gates, five operational scenarios, and independent finite gate answers remain the contract. Public operations and checkpoints belong to this application. Native loading and source-call inspection do not prove learned execution quality. Effect intent is distinct from an authorized receiver receipt.

The public operation constructs its complete typed carried state on every return, including blocked and no-change outcomes. Only the domain-specific observed change updates its own fields; prior receipts, unrelated history and unresolved external outcomes remain inspectable. Guardrail findings are consumed by the result or the action eligibility they govern.
