---
args: { request: CloseCycleRequest, reports: EvidencePage[] }
returns: CloseGuardrail
---
Compare source report content hashes and line-level changes, even when regenerated totals are identical. Preserve both versions and invalidate workpaper assertions that consume changed rows.
