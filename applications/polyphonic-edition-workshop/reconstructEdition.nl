---
description: Reconstruct all source-supported ensemble editions from an immutable evidence directory.
kind: directory-reducer
args:
  request: EditionRequest
returns: EditionCertificate
---
1. Call readArchive(request) on this folder and consume its actual original source snapshot and typed
   role, convention, partbook, editorial and rehearsal contributions. Keep source uncertainty explicit.
2. Call buildFamilies(archive.roles). Its actual copying families describe dependence, not priority.
   Call arbitrateReadings(archive, families) and retain its actual per-location alternatives and exclusions.
3. Call assembleEditions(readings). Each returned hypothesis makes consistent source-level editorial
   choices. A later performance reprise reuses that choice rather than creating a new independent choice.
4. Call performEditions(editions, archive.protocol). Consume the actual stateful timelines, including
   provenance and any unresolved notation effect; never assemble a preferred score directly here.
5. Call checkEnsemble(performed, archive.rehearsal, archive.protocol). Pass its actual checked editions
   to explainAlternatives with archive.editorial. Retain rejected and unresolved branches separately.
6. Call auditEdition(request, archive, families, readings, editions, checked, ambiguities). Return that
   actual certificate. This composition owns neither event interpretation nor a hidden tie-breaker.
