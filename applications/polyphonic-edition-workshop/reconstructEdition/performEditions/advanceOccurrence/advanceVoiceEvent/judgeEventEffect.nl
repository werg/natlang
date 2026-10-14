---
description: Judge one notation effect from the actual selected reading and carried state.
args:
  option: ReadingOption
  state: NotationState
  protocol: NotationProtocol
returns: EventEffectDecision
---
1. Bind sourceEvent=option.event, before=state and the actual protocol. Require sourceEvent.voiceId
   to equal before.voiceId; retain an identity mismatch as an unresolved EventEffectDecision. This
   invocation owns exactly one event effect and consumes no later event or expected performance.
2. Declare the supplied EventMeaning and EventEffectDecision shapes. Bind the actual immutable protocol
   in a scoped literal; its varying input is this option and before state. Directly await this judgment:
   const judgeEffect = nl.with<EventEffectDecision>({protocol})`1. Match option.event.durationGlyph to the supplied duration lexicon. Keep a missing or conflicting glyph meaning unresolved with meaning=null.
   2. Read the actual event kind. A rest consumes the declared duration, produces no effectiveWrittenRank and preserves accidentals. For a pitched event require a source-established rank; do not infer it from the desired ensemble sound.
   3. Copy the actual before.accidentals as the candidate after-state. Find the entry for this event's written rank; an absent entry means the source-defined unaltered setting. Interpret protocol.accidentalPolicy to decide whether raise replaces that rank's setting, natural cancels it, or none consumes the carried setting. Preserve every unrelated rank entry.
   4. For an attack compute effectiveWrittenRank from this event's written rank and the just-interpreted setting. Read the source-selected option already containing any justified editorial pitch shift; that shift is not a second accidental to apply again.
   5. For a continuation read the actual previousSoundingRank and the actual voice transposition. Preserve the prior sounded rank only when the source continuation convention establishes it; derive its pre-transposition rank and return attacks=false. Missing previous tone or conflicting written continuation remains unresolved. A repeated attack is not automatically a continuation.
   6. Return one EventEffectDecision. For a complete meaning include the actual duration, pre-transposition effectiveWrittenRank, attack flag, candidate nextAccidentals, a reason naming the consumed state and nonempty exact option/protocol witnesses. For unresolved meaning keep the precise missing premise. The parent applies instrument transposition once after consuming this result.`;
   const decision = await judgeEffect({option,before});
3. Consume the actual decision. For meaning=null preserve the child's unresolved premises and return
   it directly; do not materialize a numeric note merely to let the frontier advance.
4. For a complete meaning verify evidence is nonempty and each exact quote belongs to actual option
   or protocol witnesses. Require integer positive duration and unique nextAccidentals rank keys.
   Those checks validate the returned representation; the child owns the accidental interpretation.
5. Confirm the returned reason identifies the relevant source policy or carried setting. Preserve
   unknown findings accompanying an otherwise complete reading; do not silently discard uncertainty.
6. Return this actual EventEffectDecision. Keep before immutable. Only the parent can commit its
   nextAccidentals, elapsed-time update and performed event into the changing voice frontier.
