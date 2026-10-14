---
description: Check actual performed editions against scoped independent ensemble constraints.
args:
  performed: PerformedEdition[]
  rehearsal: RehearsalIndex
  protocol: NotationProtocol
returns: CheckedEdition[]
---
1. Initialize checked empty. For each actual PerformedEdition bind events and original hypothesis;
   start findings empty and preserve its unresolved notation issues. No expected timeline enters this stage.
2. For each actual TimingConstraint select every source-defined matching occurrence. Gather the actual
   performed events in that occurrence; call judgeConstraint(constraint, occurrenceEvents) and await it.
3. For scope applicable=null retain a null finding with its actual missing premise. For false retain
   the explicit nonapplicability reason without treating that observation as a satisfied numeric relation.
4. For applicable=true validate selected original event IDs against occurrenceEvents and consume the
   actual scope reason. The following arithmetic evaluates that established relation, not a new scope.
5. For section-duration, collect each represented voice's actual onset/end interval. Compare its last
   end minus occurrence start with the source ticks, and retain a per-occurrence failed finding if any
   source-required voice or event is missing. A different voice's correct length cannot fill the omission.
6. For same-onset, collect the selected events' actual onsets in this occurrence and require equality.
   Preserve all actual values and source witnesses in the finding. A sustained note sounding at that
   time is not a new attack unless the interpreted observation explicitly describes sounding overlap.
7. For end-sonority, identify the sounds active just before the source section endpoint: onset is
   strictly earlier than the endpoint and end is at least that endpoint. Exclude actual rests and
   compare the unordered multiset of sounding ranks with the supplied observed multiset. Do not assign
   observed ranks to voices unless the actual scope judgment establishes those owners.
8. Independently visit each required voice in every occurrence. Order its actual events by onset and
   check continuity from the occurrence start to the protocol section endpoint, with every chosen source
   event represented once. Zero events, gaps, overlap or wrong order produce explicit failed findings.
9. Set admissible=false if any established constraint fails. Otherwise use null when notation/scope
   uncertainty or missing contribution remains; use true only when actual complete events satisfy all
   governing constraints. Preserve every failed/null finding rather than silently dropping the edition.
10. Append CheckedEdition with the actual performed edition and findings. Return all checked editions
    for final audit; explanation of ambiguity compares the actual admissible set and makes no new score.
