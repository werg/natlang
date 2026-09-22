/*---
engine: typescript-host
args:
  selected: string[]
  collection_revision: string
returns: Passage[]
---*/
return host.evidence.read(args.selected, args.collection_revision);
