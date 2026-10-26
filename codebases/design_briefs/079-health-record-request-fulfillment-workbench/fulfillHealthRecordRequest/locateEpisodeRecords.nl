---
description: Retrieve records only from the authorized patient, category and episode window while reporting coverage gaps.
args: { dossier: RecordsDossier; change: FulfillmentChange; authorization: ReleaseAuthorization }
returns: RecordSelection
---
Search the authorized repository using patient identity, category and date window. Bind every selected document to an episode and source reference. Do not include unrelated history to make the packet appear complete. Report each searched source and missing coverage. If the identity or episode binding is ambiguous, retain the record as unresolved.
