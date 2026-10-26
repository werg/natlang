---
description: Decide field-level findings for releaseAccessible.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Reconcile reviewer feedback, maintain side-by-side source mapping, and publish the accessible edition with its scope and version. Revalidate affected meaning units after source changes and preserve access to authoritative wording where required. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
