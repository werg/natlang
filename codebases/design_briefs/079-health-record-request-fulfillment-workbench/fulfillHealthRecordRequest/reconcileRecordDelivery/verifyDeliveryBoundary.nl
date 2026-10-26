---
description: Recheck release authorization and exact packet approval at the moment of delivery.
args: { dossier: RecordsDossier; change: FulfillmentChange; authorization: ReleaseAuthorization; packet: RenderedPacket }
returns: ReleaseAuthorization
---
Recheck expiry, recipient, patient, requested categories and date window against current authorization. Compare the exact rendered artifact hash with reviewer approval and metadata scan receipts. If any binding differs or authorization expired, block dispatch and preserve the proposed artifact without releasing it.
