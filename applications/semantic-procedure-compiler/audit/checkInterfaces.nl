---
description: Judge child/parent state ownership and preserve source ambiguity through generated callable interfaces.
args:
  program: GeneratedProgram
  schema: StateSchema
returns: InterfaceFinding[]
---
1. Inventory actual module call edges, typed arguments/results and owned semantic fields from
   generated bodies/plans. Match each child path to its actual emitted module and type declaration.
2. For each edge compare what the caller passes with what the child needs. Verify original actor,
   source identity, event time and unknown state have typed destinations rather than prose-only warnings.
3. Confirm returned child/literal fields have concrete consuming branches or state updates in the
   parent. A final output field cannot conceal an unused earlier decision and a re-derived answer.
4. Preserve nullable findings where the source remains unresolved. A parent's boolean readiness slot
   cannot represent a child's unknown evidence without an explicit typed uncertainty alternative.
5. Return one InterfaceFinding per relevant caller/callee/field with preserved true,false or null,
   precise reason and source witnesses. Exact signatures/path validation is separate from these semantic joins.
6. Keep genuine independent branches independent: custody, readiness and queue status remain distinct
   fields even where a source document names the same actor in all three clauses.
