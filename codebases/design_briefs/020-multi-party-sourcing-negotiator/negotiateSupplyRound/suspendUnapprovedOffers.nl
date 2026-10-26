---
args: {"book": "NegotiationBook", "mandate": "MandateBoundary", "evidence": "EvidencePage[]"}
returns: "SupplierEnvelope[]"
---
For every outstanding envelope compare its mandateVersion with the new mandate and its binding receipt evidence. Suspend unsent drafts. For sent or unknown envelopes preserve actual external status and request a scoped withdrawal or confirmation; do not label cancellation complete from local suspension. Retain accepted commitments for owner review under their actual signed authority, and return each envelope with its original operation ID.
