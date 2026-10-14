---
description: Interpret emitted behavior independently, then compare a separate source-required judgment.
args:
  event: ScenarioEvent
  atoms: StateAtom[]
  sourceAtoms: StateAtom[]
  branch: BranchPlan
  model: ProcedureModel
  module: GeneratedModule
  program: GeneratedProgram
returns: GuardDecision
---
1. Prepare emittedContext from the actual module text and generated callee/type-file closure in
   program as exact {path,text} records, excluding ModulePlans/contribution audits. Include actual
   atoms and original event evidence. Supply selected clause definitions only as
   runtime data when the emitted body explicitly reads them; do not supply ideal BranchPlan guidance,
   sourceAtoms, expected decisions or model.schema's normative invariants as interpretation rules.
   The actual body controls which runtime data are read and which calls/effects occur.
2. In eval declare StateEffect as {key:string;subject:string;value:string|number|boolean|null;
   known:boolean;establishedAt:number|null;witnesses:{sourceId:string;path:string;quote:string}[]}.
   Declare EmittedJudgment with value:boolean|null, effects:StateEffect[], reason:string,
   generatedSpans:{path:string;quote:string}[], witnesses:{sourceId:string;path:string;quote:string}[]. Construct a
   scoped NL emitted-behavior literal over emittedContext and directly await it on event.
3. Its instructions are: symbolically interpret the actual emitted guard/effect wording against
   this event and actual state, following actual generated child bodies in the supplied code closure;
   preserve unconditional permissions, wrong clause selection or lost prerequisites exactly as emitted.
   Missing actual callee/data is unresolved rather than guessed from the ideal source behavior. Return
   resulting effects and exact path-keyed generated text spans from the supplied code closure,
   without repairing the body.
4. Consume its generatedSpans as GeneratedSpan evidence. For each span require its path to identify
   an actual supplied emittedContext code-closure record and its nonempty quote to occur exactly in
   that record text. Preserve a child-file span at that actual child path. Require at least one
   nonempty exact generated span for every decided emitted
   permission or nonempty effect collection. Missing or invalid support leaves that emitted decision
   unresolved and cannot certify a match. Keep original event/state quotations in the original Witness
   domain; emitted code is not an original source record.
5. Separately prepare requiredContext with the governing branch, selected source clauses/exceptions,
   sourceAtoms and original event evidence. Declare RequiredJudgment with value:boolean|null,
   effects:StateEffect[], satisfiedPremiseIds:string[], missingPremiseIds:string[], waivedPremiseIds:
   string[], reason:string and witnesses:{sourceId:string;path:string;quote:string}[]. Construct a
   second scoped NL source-requirement literal and directly await it on the same original event.
6. Its instructions are: judge active source-required prerequisites at this actor/event time; apply
   only source-scoped waivers; retain custody/readiness independence, station-global hazard ownership,
   false versus unknown, and the source-required state effects. The actual emitted decision is excluded
   from its context, and it cannot inherit a fabricated atom from the emitted-state channel.
7. Consume emitted.value/effects as GuardDecision.value/effects; consume required.value/effects as
   sourceRequiredValue/sourceRequiredEffects. Compare returned permissions, actor/key/time effects
   and known/unknown distinctions, preserving semantic equivalence rather than prose byte equality.
   Return matchesSource=false for an established divergence or null for a missing decisive comparison.
8. Record distinct actual local decisions for the two literal results. Preserve generatedEvidence
   separately from original witnesses. A failed generated interpretation remains in the emitted trace,
   while its source-required judgment supplies the counterexample repair obligation.
9. Return ruleId, eventId, both value/effect channels, matchesSource, actual premise distinctions,
   original witnesses, generatedEvidence, reason, localDecision and sourceLocalDecision. Parent replay
   applies each effect collection only to its own state channel; neither can silently correct the other.
