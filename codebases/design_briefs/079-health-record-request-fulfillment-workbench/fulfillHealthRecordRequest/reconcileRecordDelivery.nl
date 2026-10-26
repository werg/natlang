---
description: Reconcile authorization, exact artifact approval and recipient receipt before recording disclosure.
args: { dossier: RecordsDossier; change: FulfillmentChange; authorization: ReleaseAuthorization; packet: RenderedPacket }
returns: RecordDelivery
---
Call `verifyDeliveryBoundary(dossier, change, authorization, packet)` immediately before dispatch. Require authorization still current and approval bound to this packet's content hash. Record delivery only from a recipient receipt bound to recipient, artifact, time and channel. If a corrected record arrives after an earlier disclosure, create an amendment and notification obligation linked to that delivery; preserve the earlier packet unchanged.
