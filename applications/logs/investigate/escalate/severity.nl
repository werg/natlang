---
description: Decide how severe an incident is.
readout: decision
args:
  claim: string
  evidence_count: number
  services: string[]
returns: Severity
---
Decide the severity of a problem described by claim, supported by evidence_count log records, affecting services.

- low: a degradation few users would notice.
- medium: a repeated failure of one feature.
- high: a failure of an important service, or of several features.
- critical: data loss, a security breach, or a service fully down.
