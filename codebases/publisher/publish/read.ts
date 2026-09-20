/*---
engine: typescript-host
args:
  span_ids: Text[]
  collection_revision: Text
returns: Passage[]
---*/
return host.publisher.evidence.read(args.span_ids, args.collection_revision);
