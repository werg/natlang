---
args: { prepared: PreparedInvoiceMatch, goal: GoalContract, policy: PolicySelection }
returns: DecisionsInvoiceMatch
---
1. Apply this domain algorithm to each relevant evidence unit: Align descriptions, units, supplier references, and delivery context across invoice and order lines. Permit many-to-many matches for split shipments, but preserve ambiguous allocations and do not force every invoice line onto the nearest description.
2. Return field-level atoms for the actual InvoiceMatch record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
