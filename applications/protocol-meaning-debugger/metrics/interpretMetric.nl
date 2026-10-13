---
description: Read symbolic numerator wording, denominator scope and displayed label before raw-code binding.
args:
  ownerId: string
  clause: DefinitionClause
  question: Question
returns: MetricReading
---
1. Read the selected clause as the implementation's actual instruction. Separate the wording for
   what it counts from the wording used to display the result and from the target's desired question.
2. In eval declare MetricReadingJudgment with countedTerm: string | null, denominator: 'attempts' | null,
   aggregation: 'pooled-percentage' | null, displayLabel: string | null, reason: string, witnesses:
   {sourceId:string;path:string;quote:string}[]. Capture clause and question as the local context.
3. Construct nl.with<MetricReadingJudgment> over the selected clause wording. Its instructions are:
   extract the counted proposition's symbolic term, denominator events, pooling rule and displayed
   label; keep conflicting or unstated roles null and cite the actual controlling phrases. Read the
   implementation as written even where its display language differs from the counted proposition.
4. Directly await the literal. Consume its countedTerm, denominator, aggregation and displayLabel
   in MetricReading. The child's task is a symbolic source reading; the parent binds the counted
   term to source-defined operational codes in a separate judgment.
5. For each required null add an unresolved finding naming the missing role and selected clause ID.
   A target definition can clarify terminology but cannot replace the implementation's instruction.
6. Record localDecision with this site, owner and clause ID, the actual returned role distinction and
   its exact witnesses. Retain the literal's actual output, including a misleading display phrase.
7. Return implementationId, definitionId, countedTerm, denominator, aggregation, displayLabel,
   witnesses, localDecision and unresolved. Parent code-binding consumes this complete typed reading.
