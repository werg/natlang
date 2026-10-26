# Purchasing conformance design

The desk resolves the requested function and quantity before looking for an agreement. It matches the purchasing legal entity, effective dates, territory, schedule, and amendments explicitly; an affiliate is covered only when the agreement says so. Each line is checked separately for functional scope, price, bundle conditions, quantity minimums, and use restrictions. A changed catalog description stays tied to its observed revision. Removing a discounted bundle item triggers repricing against the remaining minimum. Related commitments aggregate before threshold routing, and expired terms block a pending release until the exact dependencies are refreshed.

The result is a review and approval proposal. It does not create a purchase order or release an order.

If essential intent facts are missing, the operation still returns its complete typed review: agreement, conformance, options, and disposition are explicitly null, independent guardrails are retained, and the checkpoint records the unresolved intent. No supplier item is selected on that path.
