---
args: {"docket": "BillingDocket", "slices": "BillableSlice[]", "acceptance": "AcceptanceDecision[]", "prior": "ObligationBook"}
returns: "MilestonePacket"
---
Create one customer explanation per proposed milestone/phase amount, naming its contract condition, acceptance evidence, effective rate schedule and prior consumption. Exclude held slices from invoice totals while preserving them as open obligations. Bind approval to the exact source version, phase identities, amount, customer and invoice operation identity. Any late source change invalidates only affected slices and their packet approval. Preserve authoritative wording where a short explanation would widen scope. Return no ready packet when any included slice lacks a current condition or exact remaining capacity.
