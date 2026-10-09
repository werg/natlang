---
description: Migration report, closing words. What changed, whether the checks pass, and what a reviewer should look at.
args:
  intent: Intent
  plan: Plan
  edits: SiteEdit[]
  state: RepairState
  report: MigrationReport
returns: Summary
---
Write the closing words of the migration for a reviewer.

- summary: at most three sentences. Say whether the checks pass (report.status), how many files changed
  (report.changed) and how many sites were edited and left (edits by status), and the intent in a few words.
- next: one short sentence each for what to review: sites left alone that may matter (plan.risks, edits left or
  failed), the findings still open (state.findings that are not repairable, or any when the checks fail), and
  state.rejected when it is not empty. Empty when the checks pass and nothing was left that matters.

Return { summary, next }.
