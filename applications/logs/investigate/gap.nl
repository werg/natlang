---
description: Source gaps. Say which open incidents a gap in the log source touches, and what is now unknown.
args:
  item: LogEvent
  incidents: Incident[]
returns: GapNote
---
item is a gap: records are missing from the source around item.occurred_at, and item.message says why.

1. affected: the IDs of incidents whose time span (opened_at to closes_at) includes item.occurred_at.
2. unknown: one sentence, "Source gap at cursor N: <message>", with N the item's cursor, adding which incidents'
   evidence may be incomplete.
