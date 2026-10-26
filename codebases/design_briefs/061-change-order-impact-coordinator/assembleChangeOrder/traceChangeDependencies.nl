---
args: { scope: ScopePartition, edges: ScopeDependency[], evidence: Evidence[] }
returns: DependencyTrace
---
Bind findings = edges.map(edge => qualifyDependencyEdge(scope, edge, evidence)). Start frontier with scope.affectedAreas and an empty reachable package set. Repeatedly add a supported edge's targetPackage only when its sourceArea is already in frontier; then add that package to frontier so a later supplied edge may continue the chain. Stop when a pass adds no new package. Never traverse cycles twice. Retain each traversed edge with its evidence reference, and return unresolved edges that originate on a reached path. Unsupported links cannot carry the path to another package. Return unique reachable package IDs with both evidence sets.
