---
description: Decide whether a pull request may merge given the grounded review findings.
args:
  findings: Finding[]
returns: MergeVerdict
readout: decision
---
Decide the merge verdict from findings:
- "block" when any finding is a security, data-loss, or correctness blocker that must not ship;
- "request-changes" when findings need edits before merging but do not pose an immediate safety block;
- "approve" when there are no findings or only minor optional suggestions.
