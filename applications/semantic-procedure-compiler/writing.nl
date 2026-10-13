---
description: Author a typed callable NL program from the actual module plans and semantic branch ownership.
args:
  plans: ModulePlan[]
  model: ProcedureModel
  request: CompilerRequest
returns: GeneratedProgram
---
1. Inventory the finite returned plans and their call graph. Validate relative paths and unique
   ownership exactly; expose a missing planned child instead of substituting an unrelated available stage.
2. Define generated type files from model.schema and ModulePlan interfaces. Preserve original fact
   identity, owner, establishedAt, unknown state and source witness fields through every child boundary.
3. For each plan call writeModule(plan, model, plans). Consume its actual text and contributionAudit.
   Keep policy and semantic branch selection as numbered NL in the generated callable tree.
4. Inspect every returned audit. A module without a new contribution or with an unconsumed literal
   is a writing defect requiring repair, not a successful way to increase declared depth.
5. Construct sourceToModule from each selected clause's actual rule/module responsibility. Historical
   versions and exceptions can map to the same reusable guard with distinct captured source context.
6. Return entryPath, modules, typeFiles, sourceToModule and unresolved. Scenario execution must inspect
   these actual generated bodies/plans; replaying only the compiler's preferred source model would
   fail to detect a module that emitted a different guard or dropped a carried state field.
7. Keep generated source as immutable proposal data. Exact compiler/type checking belongs at the
   outside-world seam and cannot certify its semantic equivalence to a procedure clause.
