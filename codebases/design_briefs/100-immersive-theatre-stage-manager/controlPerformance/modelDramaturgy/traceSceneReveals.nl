---
description: Trace a directed scene graph of prerequisite facts and controlled reveals.
args: { production: ImmersiveProduction; sceneIds: string[]; showEvidence: EvidenceItem[] }
returns: SceneRevealGraph
---
1. Bind interpreted = extractRevealEdges(sceneIds, showEvidence). Retain the full production.sceneIds as the named scene universe, including prerequisites outside the currently affected scene set. An event's sceneIds are a scope of review, not proof of the performance order.
2. For each interpreted edge, bind the exact source-established originating scene, dependent scene, fact, characters and audience groups. Report absent scene identities in gaps. Keep disputed and missing edges inspectable; they cannot establish a permitted reveal.
3. Traverse each documented route from its established entry scenes. Before a fact is used, establish a preceding permitted reveal for that recipient on that route. Record a blocked route when the evidence establishes a cycle, use before disclosure, or a crossing of a protected recipient boundary. Record an unknown ordering in gaps rather than deciding order from array position.
4. Derive sceneOrder from the current approved route and prerequisite relations, preserving the documented order of independent scenes. If alternative routes differ, retain their distinctions on the edges and report only an order supported for the current review. Return SceneRevealGraph { productionId: production.productionId, sceneOrder, edges: interpreted, blockedRoutes, gaps, sourceIds: exact sources used }. Do not erase a rejected edge merely to obtain an apparently acyclic graph.
