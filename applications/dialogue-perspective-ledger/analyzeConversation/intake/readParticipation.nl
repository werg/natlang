---
description: Interpret channel and private-note access without inferring belief.
args:
  clauses: SourceClause[]
  turns: TurnRecord[]
returns: ParticipantIndex
---
1. Select actual participant/channel definitions and their stated validity intervals. Read changes to
   membership as separate witnessed intervals, rather than treating a channel label as perpetual access.
2. Construct and directly invoke a scoped NL ParticipantIndex literal over those originals. Its local
   instructions identify named actors, explicit membership and private-note ownership, then report unknowns.
3. Consume the returned participantIds and memberships. Exact source actor spellings are retained;
   generated role descriptions are free prose and must not supply an unstated access privilege.
4. Check each turn speaker/audience against this original actor domain. A role such as coordinator is
   not evidence of honesty, omniscience or reading an attachment. Keep unresolved references explicit.
5. A recipient outside a channel may still gain access through an actual directed receipt; membership
   alone does not prove a particular turn was heard. Those joins belong to joinAvailability.
6. Return the interpreted index with original witnesses. Do not classify any participant's belief here;
   that requires the later actual endorsement decision in the participant's own episode.
