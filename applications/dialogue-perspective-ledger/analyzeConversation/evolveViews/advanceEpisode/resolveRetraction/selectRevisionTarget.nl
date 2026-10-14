---
description: Resolve an actual performed or adopted revision to its original target lineage before owner-specific effects.
args:
  act: ActReading
  decision: EndorsementDecision
  episode: Episode
  source: DialogueSource
  scoped: ScopedConversation
returns: RevisionSelection
---
1. Preserve the actual current act, endorsement decision and episode horizon. Inventory original
   correction/retraction candidates in scoped whose original event is at or before this episode.
   A quoted depiction retains its attributed original but cannot itself issue another revision.
2. Resolve the actual decision.adoptsActIds to supplied candidates. The current act can be an adopting
   acknowledgment rather than a performed revision. If an adopting correction also repeats the original
   content, retain the named adopted revision as its original target basis and the current actor as adopter.
3. For a current performed correction/retraction not adopting another original, retain its own actual
   candidate. Private cancellation intention, doubtful nonadoption or report-only quotation supplies no
   applied candidate by itself. Preserve competing explicit targets rather than selecting by recency.
4. Capture the actual act/decision, selected candidate objects, original source clauses, scoped acts and
   episode. Directly await a scoped NL RevisionSelection literal on those actual typed objects. Its
   numbered local policy interprets the candidate's own stated target/exclusions, joins superseded original
   turn IDs to actual source acts, and distinguishes corrected propositions from promise cancellation.
5. The literal judges communication from original issuer/audience and actual source channel/receipt
   evidence. Communicated means established outward communication of that revision, not receipt
   or adoption by every beneficiary. A later report does not turn a private intention into a communicated revision. It returns
   kind=belief-correction for the exact corrected proposition scope, kind=commitment-retraction only
   when the original performed promise is targeted, and none for established nonrevision/nonadoption.
   Missing target meaning or consequential competing targets return unresolved with their actual premises.
6. Consume actualSelection. Check any revisionActId identifies the actual selected original candidate;
   every targetActId belongs to supplied scoped and every targetPropositionId to source.propositions.
   A commitment target must identify an original performed promise, without inventing a new target ID.
   These checks validate returned references; they do not replace the literal's target judgment.
7. Require nonempty original source support for a decided selection and exact quote membership in
   actual source clauses. Preserve clauseId/sourceId/path from parent-held originals. A none result
   retains its negative source basis; unresolved target/communication cannot become false by default.
8. Return the actual RevisionSelection, retaining original target identity, nullable communication,
   exact witnesses and missing premises. Parent resolves actual owner and audience views from this
   typed result, rather than inferring a target from whichever commitment happens to be nearby.
