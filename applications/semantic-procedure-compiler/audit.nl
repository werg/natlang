---
description: Join independent source, interface, contribution and scenario obligations for generated policy.
args:
  intake: Intake
  model: ProcedureModel
  program: GeneratedProgram
  suite: ScenarioSuiteResult
returns: CompilationAudit
---
1. Call checkInterfaces(program, model.schema). Consume actual caller/callee field-preservation
   judgments and unresolved ownership gaps. A compatible signature alone cannot prove propagated meaning.
2. For every selected source clause inspect its sourceToModule mapping and actual governed rule/branch.
   Establish sourceComplete only when its behavior and historical/exception scope are represented.
3. Inspect module contribution audits and declared literal consumers. A module can be substantively
   authored while live execution remains unmeasured; source contributionComplete refers only to this audit.
4. Inspect all requested original scenarios and their actual returned traces/verdicts. Set
   scenarioComplete only when every source-supported required observable is satisfied or explicitly unresolved.
5. Combine independent findings without letting one success overwrite another failure. The generated
   application can be well-typed and still lose safety, actor ownership, uncertainty or temporal direction.
6. Return interfaces, sourceComplete, scenarioComplete, contributionComplete, findings and unresolved.
   These source/semantic audits are candidate evidence; actual live stage consumption remains separately graded.
