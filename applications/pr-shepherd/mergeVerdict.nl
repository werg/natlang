---
description: Decide whether a pull request may merge given the grounded review findings.
args:
  findings: Finding[]
returns: MergeVerdict
readout: decision
---
Decide the merge verdict from findings:
- "block" when any finding is a security, data-loss, tenant-isolation, or critical correctness blocker that must not merge;
- "request-changes" when findings require author edits or missing tests or docs before merging, without an immediate safety block;
- "approve" when there are no findings or only minor optional suggestions.
