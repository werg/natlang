---
description: Decide field-level findings for linkCampaigns.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Detect copied or campaign-template language while preserving independently added reasons and legitimate participation counts. Distinguish duplicate ingestion from multiple people submitting the same position, and report volume and argument diversity separately. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
