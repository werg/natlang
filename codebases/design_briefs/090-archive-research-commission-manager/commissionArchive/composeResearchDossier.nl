---
description: Compose a reproducible dossier and source-revision checkpoint.
args: { request:CommissionRequest; sources:ArchiveSource[]; observations:SearchObservation[]; leads:FollowupLead[]; claims:SupportedClaim[]; checkpoint:ResearchCheckpoint }
returns: CommissionDossier
---
For every question state the supported finding or unresolved disposition. List searched, closed, restricted, unsearched and exhausted collections distinctly. Preserve citations, source revisions, interpretation notes, rights restrictions and remaining leads. Do not reproduce restricted material; offer its authorized citation or description. Return a checkpoint that reopens only claims dependent on changed source revisions. Set extension-needed when a valuable open lead exceeds remaining effort.
