---
args: {"evidence": "EvidencePage[]", "packages": "PackageCandidate[]", "mandate": "MandateBoundary"}
returns: "TermReply[]"
---
For each response call separateReplyTerms(response, packages). Match package and term identity, signer, mandate and source version. A reply modifying delivery while accepting price accepts only that price conditionally; record its modified delivery as a qualified term. Retain verbal acceptance as unconfirmed unless the configured mandate accepts that evidence. Deduplicate repeated messages by source identity, not by similar wording.
