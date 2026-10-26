---
description: Derive concrete failure cases from graph states and declared props and routes.
args: { room: EscapeRoomState; proof: PuzzleVerification; variants: AccessibilityRoute[] }
returns: FailureCase[]
---
For each proof state, create one lost-prop case for each prop required by a node that can still progress from that state. Create a repeat-action case for each enabled node, because repeated solving must preserve the same world state. For each clue whose node is not yet enabled but whose content could be discovered early, create an out-of-order case tied to that state and node. For each declared accessibility route, create a case at the earliest state where its equivalent clue path diverges from the default path. Bind every case to the proof's exact state ID and affected node or prop IDs. Do not create a case for an unobserved or unreferenced prop.
