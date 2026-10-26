---
description: Prepare an episode-bounded health-record packet under a current recipient-specific authorization.
args: { dossier: RecordsDossier; change: FulfillmentChange; officer: RecordsOfficer }
returns: FulfillmentProposal
---
Bind the request to requester, patient, authorized recipient, categories, date window and expiry. If authorization does not cover this request, stop before rendering or delivery.

Bind authorization = `qualifyReleaseAuthorization(dossier, change)`. Bind selection = `locateEpisodeRecords(dossier, change, authorization)` only within authorized patient and episode scope; keep unrelated history out and report coverage gaps.

Bind reviews = `reviewResponsiveDocuments(dossier, change, authorization, selection)`. Separate responsive spans from third-party information and route only affected spans for qualified review.

Bind packet = `renderAndSanitizePacket(dossier, change, authorization, selection, reviews)` only after disclosure review. Verify visible content and hidden metadata on the exact rendered artifact, then bind approval to its hash and authorization version.

Bind delivery = `reconcileRecordDelivery(dossier, change, authorization, packet)` only after all checks pass. Recheck authorization immediately before dispatch and require an exact recipient receipt. If a corrected record arrives later, append a controlled amendment and notification obligation; never mutate the prior packet.

Return `FulfillmentProposal` for `officer.id` with authorization scope, episode coverage, targeted review results, artifact identity, delivery state and resumable amendment history. A prepared artifact is not a disclosure until the authorized recipient receipt is recorded.
