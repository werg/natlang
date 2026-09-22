/*---
engine: typescript-host
args:
  span_ids: string[]
  collection_revision: string
returns: Passage[]
---*/
return host.publisher.evidence.read(args.span_ids, args.collection_revision);
