---
description: Join original event-time units and actual availability into a finite ordered plan.
args:
  source: DialogueSource
  scoped: ScopedConversation
returns: EpisodePlan
---
1. Inventory original utterance/private-note events, each actual received/read-content event and each
   physical log event. Dispatch-only events can be traced but cannot activate recipient availability.
2. From actual performed act conditions and due fields inventory source-established closure boundaries.
   A strict before cutoff closes at its stated moment; a day-only unconditional due closes when that day
   ends, preserving unknown due hour. Represent these as kind=boundary with original rule/act witnesses,
   no invented utterance, receipt or physical event and no new recipient availability. They permit typed
   condition/deadline reevaluation even when no one speaks at that moment.
3. Create generated episode identities tied to original IDs and established source moments. A historical
   physical occurrence and a later receipt of its report are separate episodes with separate purposes.
4. For each candidate episode call joinAvailability(episode, source, scoped). Copy its actual participant
   access rows and exact receipt/turn witnesses. A global physical fact has no recipient unless a source
   hearing or verified attachment explicitly supplies that availability.
5. Order by actual day/hour pairs. Resolve simultaneous ties only through an explicit source relation;
   independent simultaneous availability may apply as one episode with no invented causal order.
6. Keep unorderable source units unresolved with their original IDs. Do not silently date an unknown
   quote at the forward event or manufacture a receipt based on filename or ID spelling.
7. Require every actual scoped act to join its original event or attributed original quotation context.
   Episode.turnIds identify actual new utterance/private-note events at that moment. Later received content retains its original act identity in actual availability rows, without a new authored turn or replayed original speaker. A quoted historical act is not replayed as a new promise at every forwarding episode.
8. Return the actual finite EpisodePlan. It preserves original utterances and delayed receipts as
   independently witnessed units, ready for state reduction without new dynamically invented events.
