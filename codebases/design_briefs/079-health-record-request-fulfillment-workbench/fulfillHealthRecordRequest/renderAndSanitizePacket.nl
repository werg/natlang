---
description: Render an episode-bounded packet and verify its content and hidden metadata on the exact derivative.
args: { dossier: RecordsDossier; change: FulfillmentChange; authorization: ReleaseAuthorization; selection: RecordSelection; reviews: DocumentReview[] }
returns: RenderedPacket
---
Render only documents and spans cleared for disclosure. Call `scanRenderedArtifact(dossier, change, authorization, selection, reviews)` on the rendered derivative after conversion; inspect visible content and hidden metadata as separate surfaces. Bind the scan to the artifact hash. Require reviewer approval for the same hash and current authorization version. If post-render sanitization changes the bytes, render and scan again before approval.
