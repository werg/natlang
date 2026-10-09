---
description: Conflict detection. Decide how two changes to one block relate.
readout: decision
args:
  a: Change
  b: Change
returns: Relation
---
Decide how change a and change b, two updates to the same block made from the same base text, relate.

- compatible: the block can say both at once. They touch different statements, or they add different things to the same
  statement, or one only respaces.
- redundant: one of them already carries everything the other adds and removes.
- contradictory: the block cannot say both. They give different values for the same fact, one removes what the other
  builds on, or they ask for opposite behaviour.

Compare adds and removes first, and read intent and text when those are not enough to tell.
