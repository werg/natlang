---
description: Interpret independently checked dependency-plan observations.
args:
  question: string
  observations: Observation[]
returns: Assessment
types:
  Observation: '{ id: string, source_revision: string, status: string, violations: string[], minimized: string[], trace_sha256: string, detail: string }'
  Assessment: '{ confirmed_ids: string[], unknown_ids: string[], findings: string[], unknowns: string[], followups: string[] }'
---
Explain confirmed contract violations, execution failures, and unresolved cases.
Only an observation with status "violated" and a nonempty violations list is a
confirmed counterexample. A crash, invalid input, or incomplete trace is a blocker
or unknown. Do not invent an oracle or replace the supplied exact observations.
Put the ID of every violated observation in `confirmed_ids`. Put the ID of every
observation with another status besides "done" in `unknown_ids`. A "done"
observation belongs in neither list. Then explain the evidence in `findings`,
`unknowns`, and `followups`. Write one `Assessment` record with exactly these
five fields, each a list of text. Use empty lists where appropriate.
