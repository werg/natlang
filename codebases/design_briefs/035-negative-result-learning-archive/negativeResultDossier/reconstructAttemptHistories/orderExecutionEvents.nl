---
args: { planned: AttemptTimeline[], evidence: EvidencePage[] }
returns: AttemptTimeline[]
---
Join timestamped execution records to the planned step identity. Preserve attempted, omitted, repeated, and interrupted steps as distinct events; report log gaps and conflicting clock sources instead of filling them by narrative expectation.
