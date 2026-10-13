---
description: Derive carried state and causal components from actual procedure predicate needs.
args:
  intake: Intake
  requirements: RequirementSet
returns: ProcedureModel
---
1. Call inferState(requirements, intake.scenarios). Consume typed fields, owner roles, identity rules
   and source-backed uncertainty invariants. Later branches need these exact state responsibilities.
2. Call orderDependencies(requirements, schema). Consume source-backed requires/invalidates/alternative
   edges and actual component settlement, including unresolved negative cycles.
3. Join the schema with the dependency graph. For every edge verify its predicates have declared state
   keys; for every state field retain the rule that produces it or its initial source responsibility.
4. Preserve facts whose owners differ even when their names look similar: custody belongs to the
   holder, readiness to the acknowledged actor, permission to the actor at the attempted event time.
5. Expose a missing producer or temporal scope as unresolved. Lowering must not manufacture a default
   value just to complete a generated signature or erase an awkward branch.
6. Return requirements, schema, dependencies and unresolved. This model is consumed by both lowering
   and scenario execution so emitted policy cannot use a different hidden state interpretation.
