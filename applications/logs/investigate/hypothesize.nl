---
description: Hypothesis generation. Propose the explanations for an incident that the index can confirm or refute.
args:
  incident: Incident
  item: LogEvent
  runbook: string
returns: Hypothesis[]
---
Propose between two and four hypotheses for incident, the latest event being item. runbook is text the event pointed
at, empty when there is none.

1. Read item's message and level, the incident's services and codes, and any existing hypotheses (keep the ones that
   still fit, with their IDs).
2. Write each hypothesis as a claim that evidence in the log could confirm or refute: a cause (a dependency failing, a
   deploy, a resource running out), an impact (what the failures did to users or other services), or a benign
   explanation (a test, a health check, a planned restart). Include one benign hypothesis whenever the messages
   could come from ordinary operation.
3. Give the IDs "h1", "h2", and so on, in order, and set kind to cause, impact or benign.
4. When runbook is not empty, use what it says to sharpen the claims.
