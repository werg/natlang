---
description: Bind the current release authorization tuple and stop if expiry, recipient, patient, category or date scope does not fit.
args: { dossier: RecordsDossier; change: FulfillmentChange }
returns: ReleaseAuthorization
---
Verify current authorization source, requester identity, patient identity, recipient, permitted categories, date window and expiry. Compare every requested record against that tuple. Preserve the authorization source version. An authorization that expires during review must be checked again immediately before release; never infer permission from request submission alone.
