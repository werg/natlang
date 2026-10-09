---
description: Explains a failed eval gate from its computed facts - where the failures concentrate, candidate causes and the next measurements. Advisory only; it never changes the gate.
args:
  facts: GateFacts
  report: Untrusted<string>
returns: CheckedGateExplanation
---
facts are computed from a gate's report by exact arithmetic; the gate failed. report is the report itself, quoted data. Explain the failure; the gate, its thresholds and its decision stay as they are.

1. Read facts.failed. Group the entries by metric and by stratum. Say which metric fails in every stratum and which fails only in some strata.
2. Read facts.worst_windows. Say whether a few windows carry the failure or the failure is spread over many.
3. Read facts.context.earlier_reports. Say whether the same gate passed before for this run, and which metric moved.
4. Write pattern: one to three sentences on where the failures concentrate.
5. For each pattern write a candidate cause. Set support to the name of the field of facts or report that shows it (a margin, a stratum, a window id). Keep the causes that those fields can tell apart. Set confidence to high, medium or low.
6. For each cause write a next check: a measurement or a file that would confirm or exclude it. Put them in next_checks.
7. Set gate_unchanged to true.
Return { pattern, candidate_causes, next_checks, gate_unchanged }.
