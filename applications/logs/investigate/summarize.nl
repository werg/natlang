---
description: Incident summary. Write what happened, from the incident and its weighed hypotheses.
args:
  incident: Incident
  hypotheses: Hypothesis[]
  supports: Support[]
  escalation: Escalation
returns: string
---
Write the summary of incident in two to four sentences for an engineer on call:

1. What failed, in which services and codes, and since when (opened_at, last_seen as event times) with how many events
   (count).
2. The winning hypothesis (escalation.hypothesis_id) and the records that support it, by ID.
3. What is still uncertain (escalation.uncertainty) and the severity.
