---
args: {"candidates": "CandidatePage[]", "reviews": "ReleaseVerification[]", "desk": "PublicationDesk"}
returns: "ReleaseVerification[]"
---
Match supplied owner approval evidence to the exact candidate bytes, base edition, page identity, audience and required ownership set. Any changed text or changed base requiring semantic re-evaluation invalidates the old approval. A source review, request to publish or acknowledgment is not the approval itself. Return updated reviews with approvedBytes only for exact complete bindings and no semantic blocker. Missing approval leaves a bounded pending request.
