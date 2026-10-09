---
description: Decide whether an event belongs to an open incident.
readout: decision
args:
  item: LogEvent
  incident: Incident
returns: Kin
---
Decide whether item, a log event, is part of incident, an open incident.

- same: it is the same thing going wrong. It has the incident's service and code, or a service and code that the
  incident's summary or hypotheses name as connected (a database timeout and the API errors it causes), and it
  happens within the incident's time span.
- different: it is a separate matter.
