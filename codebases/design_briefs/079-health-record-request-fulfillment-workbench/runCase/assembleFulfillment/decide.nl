---
description: Decide field-level findings for assembleFulfillment.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Build a complete or explicitly partial packet with a manifest and explanation of outstanding items. Apply approved redactions through trusted rendering services, verify extracted content and metadata, and bind approval to exact artifact hashes. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
