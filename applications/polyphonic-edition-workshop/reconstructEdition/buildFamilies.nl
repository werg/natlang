---
description: Connect source copying relations without manufacturing independent support.
args:
  roles: RoleIndex
returns: FamilyIndex
---
1. Initialize FamilyFrontier with all actual original source IDs pending, the returned WitnessRole
   rows, empty families and actual role uncertainties. Preserve scope on every dependency.
2. Call advanceFamily.iterateOn with measure pendingSourceIds.length. Each step adds one source and
   its established exemplar link while retaining earlier families and original relation evidence.
3. Keep independent rehearsals separate from copied versions of those reports. A source can have
   copying dependence in one declared scope and independent testimony in another; report ambiguity
   if the returned role cannot establish the relevant scope rather than treating all text as dependent.
4. Consume the settled families. Original source ID membership is exact; authority and manuscript
   reading selection remain later decisions, independent of the generated familyId spelling.
5. Return actual families and unresolved source-dependency findings. Neither member count nor latest
   source date becomes a semantic tie-breaker here.
