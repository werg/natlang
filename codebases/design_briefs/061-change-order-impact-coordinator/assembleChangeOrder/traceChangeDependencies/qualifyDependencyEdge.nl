---
args: { scope: ScopePartition, edge: ScopeDependency, evidence: Evidence[] }
returns: DependencyEdgeFinding
---
1. Match edge.sourceArea exactly to an affected or unaffected area in scope; a source absent from both partitions is unresolved.
2. Mark supported only when edge.evidenceRef is present and its supplied evidence describes a necessary technical or contractual dependency from the source area to edge.targetPackage.
3. Set reachesUnchangedScope when the edge target is asserted to serve an unaffected area; require explicit evidence and disclose the unchanged area rather than silently enlarging change scope.
4. Return target package, source area, evidence reference, supported status, unchanged-scope flag, and the exact missing or conflicting basis.
