---
description: Judge material feasibility and goal relevance using actual typed predictions.
args:
  node: BeliefNode
  partition: ExperimentPartition
  stockRules: SourceClause[]
  goal: string
returns: ExperimentUse
---
1. Bind this node's actual stock, remaining slots and goal classes plus the supplied experiment's
   actual partition. Declare UseMeaning with feasible, goalRelevant, reason and quote support.
2. Directly await one scoped NL literal. It first interprets the source material/slot conditions and
   determines whether this experiment is executable now. A usable protocol and sufficient exact material
   are necessary; a proposed experiment label does not grant permission.
3. Next compare its actual outcome branches against the requested goal. A genotype distinction can
   be relevant without immediately solving the mechanism; a constant maternal coat observation does
   not distinguish the mechanisms merely because its wording mentions inheritance.
4. Consume the literal's result. Exact stock and original-slot checks validate the proposed judgment;
   an inconsistent check becomes an unresolved feasibility finding before transition creation.
5. Return the actual ExperimentUse with bound source witnesses. Relevance is reported separately from
   feasibility so a harmless but uninformative option can remain in the optimality comparison.
