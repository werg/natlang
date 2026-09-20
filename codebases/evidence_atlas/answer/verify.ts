/*---
engine: typescript-host
args:
  passages: Passage[]
  draft: Draft
  collection_revision: Text
returns: EvidenceAnswer
---*/
return host.evidence.verify(args.passages, args.draft, args.collection_revision);
