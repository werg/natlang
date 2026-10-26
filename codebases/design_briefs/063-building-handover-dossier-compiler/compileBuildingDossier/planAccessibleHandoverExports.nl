---
args: { profiles: AccessibleExportProfile[], deliverables: PortalDeliverable[] }
returns: string[]
---
Compare each required reader profile with the source format. If the source cannot be opened, name only a converter in the approved profile and return an accessible export task while retaining the original.
