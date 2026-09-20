/*---
engine: typescript-host
args:
  selected: Text[]
  collection_revision: Text
returns: Passage[]
---*/
return host.evidence.read(args.selected, args.collection_revision);
