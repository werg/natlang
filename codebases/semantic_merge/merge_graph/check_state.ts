/*---
args:
  state: State
returns: Bool
---*/
const ids = args.state.nodes.map(node => node.id);
const known = new Set(ids);
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && known.size === ids.length &&
       args.state.edges.every(edge => known.has(edge.from) && known.has(edge.to) && edge.relation.trim());
