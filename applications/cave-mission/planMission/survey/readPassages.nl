---
description: Distinguish directed movement, geometric adjacency and known absence in passage descriptions.
args:
  clauses: SourceClause[]
returns: PassageAssertion[]
---
1. Read every passage assertion with its survey frame and current-state qualifiers. Identify endpoint
   source IDs separately from descriptive prose, preserving local labels until frame alignment supplies
   global identities. Keep the exact endpoint keys used by the interpreted source landmark inventory.
2. Interpret physicalExistence as whether the physical connection exists, independently of allowed
   travel direction or carried-state permission. Assign present to a source-established gate, tunnel,
   corridor or chimney, absent to an explicitly missing physical connection, and unknown to unestablished
   physical existence. Preserve the exact clause supporting that particular meaning.
3. For each present one-way connection, orient from and to in the allowed movement direction. Attach
   source statements forbidding reverse traversal to that same assertion's witnesses and condition.
   Direction records its one-way permission; physicalExistence remains present for that physical gate.
4. A present two-way tunnel supplies travel in both directions subject to its carried-state condition.
   Keep a restriction on a full carrier in condition while retaining the actual tunnel's physical presence.
   Explicitly absent physical routes retain absent and their negative witnesses for later graph audit.
5. Join repeated local descriptions only when source identity and scope establish the same connection.
   Keep global and local-frame assertions available for later hypothesis-specific reconciliation. A
   source description of an outward gate and its forbidden return jointly supports one oriented row.
6. Join exact movement costs to the passage they govern, including equipment clauses defining chimney
   ascent. Preserve conditions as source-grounded language for the later current-robot-state judgment.
7. Return every relevant PassageAssertion with physicalExistence, direction, endpoints, cost, condition
   and exact witnesses. Preserve negative route evidence in these meanings and source clauses; an absence
   of physical connection, a forbidden direction and an unmet load condition remain different claims.
