---
description: Interpret metric wording under local outcome coding and target scope.
args:
  ownerId: string
  clause: DefinitionClause
  codingClauses: DefinitionClause[]
  question: Question
returns: MetricView
---
1. Capture the selected clause, coding meanings, target outcome and numerator/denominator scope.
   Treat operational outcome codes and human display names as separate references to interpret.
2. In eval declare local MetricJudgment with numeratorOutcome: 'pass' | 'fail' | null, denominator:
   'attempts' | null, aggregation: 'pooled-percentage' | null, reason: string, witnesses:
   {sourceId:string;path:string;quote:string}[]. Construct nl.with<MetricJudgment> over clause.meaning.
3. Its instructions are: identify the counted proposition, included denominator events and pooling
   rule under captured coding definitions; distinguish detecting the target signal from blocking it;
   preserve an unstated or conflicting semantic binding as null and cite the controlling clause.
4. Directly await the literal. Consume each returned field in the MetricView; add unresolved for
   any required null instead of filling it from the target protocol's different selected definition.
5. Record localDecision with this site, selected clause ID, the actual interpretation and its witnesses.
   The caller passes numeratorOutcome to the exact service, so the judgment affects real arithmetic.
6. Return implementationId, numeratorOutcome, denominator, aggregation, definitionId, localDecisions
   and unresolved. A misleading label is preserved as a diagnosable mismatch, not corrected invisibly.
7. The literal's structured result is read directly. No finite-choice scoring of the whole record or
   string equality over display labels can replace the returned semantic contribution.
