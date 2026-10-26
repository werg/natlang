---
description: Decide field-level findings for verifyDelivery.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Recheck authorization immediately before recipient-bound delivery, record actual receipt where available, and track partial fulfillment or corrections. Preserve an audit trail without exposing sensitive record bodies in operational logs or notifications. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
