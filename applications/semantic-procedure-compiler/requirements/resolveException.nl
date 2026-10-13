---
description: Judge what a source exception waives while retaining independent duties outside its scope.
args:
  exceptionClause: SourceClause
  ruleMeanings: RuleMeaning[]
  selection: EditionSelection
returns: ExceptionBinding
---
1. Gather the exception's authority, actual target wording, effective interval and affected actor/action
   predicates. Capture those meanings with the candidate prerequisites and independently stated duties.
2. In eval declare ExceptionJudgment with waivedPrerequisiteIds:string[], retainedDutyIds:string[],
   scopeMeaning:string, unresolved:string[] and witnesses:{sourceId:string;path:string;quote:string}[].
   Construct a scoped NL literal over this fixed context, directly awaited on the target rule collection.
3. Its instructions are: determine the exact scope of the waived condition, identify independent
   retained duties, and preserve an unsupported extension as unresolved. Interpret an emergency
   priority exemption as queue relief only when its source says so; keep safety/custody predicates.
4. Consume the actual waivedPrerequisiteIds into the affected rules' guard plan. Carry retainedDutyIds
   into model dependencies and scenario auditing. A waived prerequisite is not a fabricated completed
   event and does not provide its usual downstream state effects.
5. Verify returned clause/prerequisite references against the captured source/rule identities. Exact
   ID validation cannot decide whether the source's exception really entails those selected waivers.
6. Record LocalDecision with the actual changed prerequisite set, retained duties and exact witnesses.
   Return ExceptionBinding with the controlling exceptionClauseId and unresolved scope findings.
