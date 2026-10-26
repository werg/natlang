---
description: Inspect the exact rendered derivative for unauthorized content and hidden metadata.
args: { dossier: RecordsDossier; change: FulfillmentChange; authorization: ReleaseAuthorization; selection: RecordSelection; reviews: DocumentReview[] }
returns: RenderedPacket
---
Verify the artifact hash, included document list, third-party redactions, requested episode scope, and hidden metadata scan result. Any hidden identifier, unrelated history or unreviewed third-party span blocks release. Record sanitizer version and scan receipt with the exact artifact hash. Do not reuse a scan from a previous render.
