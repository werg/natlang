---
description: Target source-backed corrections in the actor who actually adopts them.
args:
  act: ActReading
  episode: Episode
  view: ParticipantView
  decision: EndorsementDecision
  views: ParticipantView[]
  source: DialogueSource
  scoped: ScopedConversation
returns: RevisionDelta
---
1. Call selectRevisionTarget(act, decision, episode, source, scoped). Name its actual returned
   RevisionSelection actualSelection. This child owns whether the current act performs a revision,
   adopts an earlier communicated one, depicts one, or supplies unresolved/nonrevision meaning.
2. Join actualSelection to the actual participant decision and episode. Resolve its revisionActId in
   scoped; an adopting acknowledgment's actor remains distinct from the original revision's issuer.
   A sender may endorse their own correction while a receiver with access alone has no adoption effect.
3. For belief-correction consume actual targetActIds and targetPropositionIds. Require this actor's
   actual access and explicit adoption or its own performed correction before selecting owned beliefs
   from those returned original target lineages. A collection-time correction cannot cancel a promise.
4. For commitment-retraction locate actual returned original promise targets in their issuer-owned
   views, separately from the adopter's own view. A fresh performed promise with another sourceActId
   is not a replacement target even if its consequent uses the same physical object or proposition.
5. For each returned promise target inspect every actual named beneficiary view. Join that actor's
   adoptedRevisionActIds and availableActIds to the same actual original revision. Current decision
   contributes only this actor's adoption; source-visible global knowledge cannot invent another adopter.
6. Interpret the source's governing withdrawal rule on those actual joins. If all required beneficiaries
   explicitly adopted the communicated revision, return only that issuer's exact commitment IDs.
   Partial adoption preserves its actual adopter set without changing global status. Missing audience
   or unresolved communication remains a missing premise, distinct from known nonadoption.
7. For none/unresolved selection retain actualSelection's negative or missing basis and no unsupported
   target effect. Existing retracted commitments remain terminal; a later true antecedent, delivery or
   forwarded report cannot reactivate or repeat a withdrawal as a newly performed promise.
8. Retain actualSelection unchanged as RevisionDelta.selection. Return a generated revisionId only
   when an actual revision effect is applied. Keep nullable owner,
   actual adoptingParticipantIds, original supersededActIds, targeted own-belief/issuer-commitment IDs,
   actual source witnesses and unresolved premises. Parent applies belief effects only to this participant
   and commitment effects only to the actual returned issuer owner, preserving before/after lineage.
