---
description: Interpret independently checked dependency-plan observations.
args:
  question: Text
  observations: Observation[]
returns: Assessment
types:
  Observation: '{ id: Text, source_revision: Text, status: Text, violations: Text[], minimized: Text[], trace_sha256: Text, detail: Text }'
  Assessment: '{ findings: Text[], unknowns: Text[], followups: Text[] }'
---
Explain confirmed contract violations, execution failures, and unresolved cases.
Only an observation with status "violated" and a nonempty violations list is a
confirmed counterexample. A crash, invalid input, or incomplete trace is a blocker
or unknown. Do not invent an oracle or replace the supplied exact observations.
Write one `Assessment` record with exactly `findings`, `unknowns`, and
`followups` fields, each a list of text. Use an empty list when a category has
no entries. Do not replace these fields with more descriptive names or notes.
