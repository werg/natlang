# 079 — Health Record Request Fulfillment Workbench

## Distinct algorithm: form a release projection from an authorization tuple and an episode-bounded record inventory

The workbench binds every requested release to requester, patient, authorized recipient, categories, date window and expiry. Authorization is checked again immediately before delivery, so a request that expires during review stops at the release boundary. Retrieval searches only the requested episode and records coverage gaps explicitly. A responsive document can still contain another person's information; targeted spans are routed for review rather than releasing the whole file or discarding it wholesale.

Fulfillment is built from a sanitized rendered derivative. A post-render scan checks document content and hidden metadata as separate channels, and a reviewer approval is bound to the exact derivative hash. Delivery is accepted only for the authorized recipient with an exact receipt. A corrected record arriving after disclosure creates a controlled amendment with a new review and notification obligation; it never silently mutates the prior packet.

The resume state preserves authorization versions, episode search coverage, source document references, third-party review decisions, rendered derivative identity, metadata scan, approval binding, exact delivery receipt, and outstanding amendment/notification obligations.
