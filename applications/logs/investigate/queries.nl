---
description: Evidence queries. Plan the searches over the exact log index that would confirm or refute each hypothesis.
args:
  incident: Incident
  hypotheses: Hypothesis[]
  item: LogEvent
  settings: LogSettings
returns: Query[]
---
Plan the index searches for hypotheses. A Query has hypothesis_id and these fields, all optional except the last three:
service, code, level (exact matches), contains (a word or phrase from the message), from, to (event times) and limit.

1. For each hypothesis, write one or two queries whose results would settle it:
   - cause: the lines of the suspected component, or of the incident's own service and code, before the first member;
   - impact: the lines of the services that depend on the incident's service, after it started;
   - benign: the lines that would show ordinary operation, such as contains "test" or "health", or a level of info.
2. from is item.occurred_at - settings.window_ms and to is item.occurred_at, unless the hypothesis needs a different
   span (incident.opened_at - settings.window_ms for a cause).
3. limit is 20.
