---
args: {"sourceId": "string", "editions": "FrozenDatasetEdition[]"}
returns: "ConsentRemoval"
---
Walk the finite released edition inventories and family memberships for this source, retaining every derived artifact and recorded recipient. Keep provenance links needed to find downstream copies without copying withdrawn attributable material into the public explanation. Match actual removal and notice receipts to their original artifact/recipient identities. Return unresolved lineage where the historical manifest lacks a required edge.
