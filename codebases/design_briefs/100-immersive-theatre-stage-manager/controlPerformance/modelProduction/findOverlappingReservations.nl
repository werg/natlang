---
description: Find exclusive-resource collisions across time-indexed scene windows.
args: { windows: ResourceWindow[] }
returns: ResourceConflicts
---
1. Bind checked = compareIntervals(windows). This exact helper checks declared UTC intervals and resource identity; it makes no allocation or authority decision. Preserve every malformed or absent interval in unverified, and every unavailable resource in unavailable.
2. For each exact overlap, inspect the source-established exclusivity. An exclusive or unknown-capacity resource cannot serve overlapping scenes. Name both scene demands and their source IDs in conflicts. Sharing is permissible only when the current production evidence explicitly establishes a sufficient shared capacity; a word such as shared alone does not establish unlimited capacity.
3. For symbolic windows lacking exact endpoints, retain the unresolved comparison and state which timing evidence is needed. An apparently different label is not proof of disjoint time. Do not choose a winning scene or silently drop a reservation.
4. Return ResourceConflicts { windows, conflicts, unavailable, unverified }, de-duplicating only identical comparisons. The exact helper's overlap observations remain evidence for the semantic resource decision.
