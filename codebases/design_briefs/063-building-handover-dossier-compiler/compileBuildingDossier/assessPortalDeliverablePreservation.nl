---
args: { deliverables: PortalDeliverable[] }
returns: DossierArchiveAction[]
---
For a portal item that expires, request a durable archive only if its permission allows copying. Count it complete only after the stored bytes have an exact supplied content hash; do not claim archive or hash generation occurred here.
