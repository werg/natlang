---
args:
  request: Request
  source: Clip
  plan: Plan
  receipt: Receipt
  inspection: Inspection
  files?: Record<string, File>
returns: Assessment
---
Assess whether this particular plan and inspected result meet the user's
meaning. If the request names a sidecar note, inspect only that files leaf. Use the exact observations, not the output filename, as evidence.
If technical inspection failed, mark intent_met false. If the request depends
on visual subject placement and visual_status is unavailable or uncertain,
set needs_visual_review true. A contradictory visual inspection must not be
called successful. Explain the evidence and any unresolved question briefly.
Write exactly intent_met, needs_visual_review, and explanation.
