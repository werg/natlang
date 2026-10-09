---
description: Incident clustering. Decide whether an event joins an open incident, opens a new one, and which open incidents are really the same.
args:
  item: LogEvent
  incidents: Incident[]
returns: Attachment
---
Attach item, a log event, to the open incidents in incidents. Work in these steps.

1. For each incident, all at once: d = decide(kin, item, incident). The incident is a "match" when d.value is "same" and
   d.confidence is at least 0.5.
2. No match: return { action: "open", incident_id: item.id, fold: [], reason }.
3. One match: return { action: "join", incident_id: its id, fold: [], reason }.
4. Several matches are one incident that was split. Take the one with the smallest opened_at as the target and return
   { action: "join", incident_id: the target's id, fold: the other matches' IDs, reason }.
5. reason is one sentence naming what the event shares with the incident (service, code, cause), or why it is
   different.
