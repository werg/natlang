---
description: Distinguish directed movement, geometric adjacency and known absence in passage descriptions.
args:
  clauses: SourceClause[]
returns: PassageAssertion[]
---
1. Read every passage assertion with its survey frame and current-state qualifiers. Identify its
   endpoints without replacing local labels with a guessed global side.
2. Separate geometry from travel permission: a physical gate can exist while reverse movement is
   forbidden. An observed surface or line of sight does not create a traversable corridor.
3. For each described passage, record kind, direction, cost, condition and existence. Preserve an
   explicit absent connection as absent; a merely unobserved connection is unknown.
4. A two-way tunnel supplies travel in both directions subject to its carried-state condition. A
   one-way water gate or upward chimney must not inherit an unmentioned reverse direction.
5. Join exact cost clauses to the passage they govern, including equipment clauses that give the
   chimney's ascent cost. Do not use geometric grid distance as a replacement for stated action cost.
6. Keep condition as source-grounded language: it will be interpreted against each current robot
   state by judgeCrossing. A narrow tunnel is not unconditionally unusable just because a later load is bulky.
7. Return every relevant assertion with witnesses, including negative evidence needed to reject
   tempting return routes. Duplicate descriptions retain witnesses rather than multiplying passages.
