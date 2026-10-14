---
description: Apply one actual NL event effect to the carried voice-local notation state.
args:
  state: VoiceFrontier
returns: VoiceFrontier
---
1. If pending is empty return state unchanged. Bind currentOption to its first actual item and before
   to the current NotationState. Preserve earlier performed events and all pending source identities.
2. Check currentOption.event belongs to this occurrence section and before.voiceId. An identity
   mismatch records this event/occurrence as unresolved; never reinterpret a different voice to fill it.
3. Directly await judgeEventEffect(currentOption, before, protocol). Retain the actual returned
   EventEffectDecision. A failed invocation keeps its concrete error and event identity; prior successful
   events remain intact. A null meaning retains the actual child's missing premise, not a default note.
4. For null or failed contribution consume this unit as unresolved with no performed event and unchanged
   before state. Later complete-certificate checks see the omitted source-event/occurrence pair; incomplete
   progress is not an admissible short performance. Continue remaining source units for independent findings.
5. For complete meaning bind effect to that actual object. Validate its nonempty exact evidence against
   currentOption/protocol and its positive integer duration. Find exactly one actual VoiceConvention;
   an absent or conflicting transposition retains a failed contribution rather than guessing zero.
6. Snapshot beforeAccidentals from before. Exact arithmetic sets onset=startTick+before.elapsedTicks,
   end=onset+effect.durationTicks and soundingRank=effect.effectiveWrittenRank+voice.transpose. A null
   effectiveWrittenRank remains a rest with null soundingRank; semantic rest classification is already owned.
7. Construct PerformedEvent from currentOption's original written identity, actual occurrence/section/
   voice, exact times, actual sounding rank, effect.attacks, the before map and effect.nextAccidentals.
   Copy actual optionId/reason/witnesses. No source-specific event array supplies any of these fields.
8. Construct nextState from before with elapsedTicks increased by this duration, accidentals equal to
   the actual effect.nextAccidentals and previousSoundingRank equal to the actual sounded rank. Preserve
   source-defined rest/continuation behavior; if the policy retains a previous tone through rest, that
   state convention must be represented explicitly before this case can be admitted.
9. Append exactly this actual performed event and remove only currentOption from pending. Return the
   changed frontier in one completion action. The parent iterateOn, not this invocation, owns the next event.
