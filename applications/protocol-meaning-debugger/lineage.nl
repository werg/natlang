---
description: Bind actual implementation roles to selected source definitions.
args:
  index: EvidenceIndex
  request: ProtocolRequest
returns: Lineage
---
1. Inventory implementation-to-definition references for the target, left and right owners. Include
   explicit revision selection, clause references and any aliases used in execution receipts.
2. For every potentially matching definition pair call resolveIdentity(leftClause, rightClause, index).
   Consume its AliasDecision rather than matching filenames or using equal prose as identity proof.
3. Follow source-supported binding chains to the population, anchor, window and metric clauses
   actually used by each owner. Preserve directed consumer-to-definition links and exact witnesses.
4. Stop a binding chain when a definition is resolved or a previously visited binding is reached.
   A cycle or missing selected revision is unresolved, not an invitation to choose an arbitrary leaf.
5. Exclude archived clauses from selectedClauses unless an explicit execution binding names them.
   An archive can explain terminology history without controlling a current implementation.
6. Record each DefinitionLink with role and implementationId. Distinct roles may use the same word
   without denoting the same proposition; retain separate links even if their text happens to agree.
7. Return links, selectedClauses, aliases and unresolved. This is a provenance contribution supplied
   to all trial materialization; trial changes do not silently change selected definition identities.
8. Repairs can change selected definition text while preserving its identity and binding. The repair
   audit rebuilds lineage to ensure an edit did not instead redirect execution to an archived artifact.
