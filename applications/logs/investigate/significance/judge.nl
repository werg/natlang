---
description: Event significance. Decide how much one log line matters.
readout: decision
args:
  item: LogEvent
  observation: Observation
returns: Significance
---
Decide how much item, one log line, matters. observation.count is how many lines of the same service and code the index
holds in the window ending at this line.

- ignore: routine. An informational record, a successful operation, a health check or test message that reports
  success, noise that repeats harmlessly.
- watch: a failure or abnormal condition worth tracking together with similar lines. The message must report
  something that went wrong.
- urgent: a failure that needs investigating at once, whatever the count: data loss, a service down, a security
  breach, or a message that says so.

Judge from the message, the level and the count.
