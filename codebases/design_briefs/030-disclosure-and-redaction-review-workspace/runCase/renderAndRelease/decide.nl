---
args: { prepared: PreparedDisclosureManifest, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDisclosureManifest
---
1. Apply this domain algorithm to each relevant evidence unit: Render irreversible redactions through a trusted host service, inspect extracted text and metadata, and assemble a manifest. Bind approval and delivery to the exact recipient and artifact hashes, retaining a controlled review copy and disclosure receipt.
2. Return field-level atoms for the actual DisclosureManifest record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
