---
description: Package generated callable NL source with typed model, scenario certificates and remaining defects.
args:
  request: CompilerRequest
  model: ProcedureModel
  program: GeneratedProgram
  suite: ScenarioSuiteResult
  audit: CompilationAudit
  revisions: RevisionTarget[]
returns: CompilePacket
---
1. Join the actual returned generated program, governed model, scenario traces, audit and revision
   targets by their source/rule/module/scenario identities. Keep their provenance and unknown fields intact.
2. Verify entryPath resolves to an emitted module, every planned child path exists and each sourceToModule
   mapping names actual files. These are exact relative-path/reference checks, not semantic approval.
3. Preserve rejected draft/habit evidence and historical governing clauses in the model's selection.
   Package source witnesses for why a generated branch changed across edition time or exception scope.
4. Collect unresolved findings without replacing them with empty success reports. A generated program
   with a demonstrated repair target remains a proposal even if its other scenarios are useful.
5. Return requestId, program, model, suite, audit, revisions and unresolved. Generated files are output
   data; the original source snapshot and external hidden canonical witness are never overwritten.
6. Name the packet's role as source/candidate preparation. Exact parser/type checks, independent
   semantic comparisons and actual live module/literal consumption are separate qualification evidence.
