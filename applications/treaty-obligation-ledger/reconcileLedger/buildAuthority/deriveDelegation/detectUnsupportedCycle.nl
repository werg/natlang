---
description: Identify unrooted citation cycles and claims whose required parent was never supported.
args:
  pending: CompetenceClaim[]
  proofs: AuthorityProof[]
  archive: Archive
returns: AuthorityRejection[]
---
1. Build the directed parent-reference graph among pending claim IDs. Keep references to absent or
   rejected parents explicit; do not convert them to roots because no local edge exists.
2. For each unassigned claim, collect forward-reachable pending IDs by repeatedly expanding a
   finite unvisited-ID frontier. Collect backward-reachable IDs through the reversed parent relation.
   Their intersection is this claim's strongly connected component; mark its members assigned and
   continue with another unassigned claim. Each frontier consumes a previously unvisited listed ID.
   A component is cyclic when it has several claims or a single claim that names itself as parent.
3. For each component determine whether an actually supported external parent supplies the required
   act and onward-delegation scope. A merely cited neighboring pending claim is not that support.
4. Return unsupported-cycle for cyclic components with no such source of competence. Return
   outside-scope for acyclic claims depending on rejected or absent parents, with their exact witness chain.
5. If a source ambiguity prevents deciding the missing root, return unresolved for that component.
   Do not settle it by repeated traversal. The caller consumes these finite components once.
