/*---
args:
  state: State
returns: boolean
---*/
const nodes = args.state.nodes;
const ids = nodes.map(node => node.id);
if (!Number.isSafeInteger(args.state.revision) || args.state.revision < 0 ||
    ids.some(id => !id.trim()) || new Set(ids).size !== ids.length) return false;
const byId = new Map(nodes.map(node => [node.id, node]));
for (const node of nodes) {
  let parent = node.parent;
  const seen = new Set([node.id]);
  while (parent) {
    if (!byId.has(parent) || seen.has(parent)) return false;
    seen.add(parent);
    parent = byId.get(parent).parent;
  }
}
return true;
