export default function check_state(state: State): boolean {
const nodes = state.nodes;
const ids = nodes.map(node => node.id);
if (!Number.isSafeInteger(state.revision) || state.revision < 0 ||
    ids.some(id => !id.trim()) || new Set(ids).size !== ids.length) return false;
const byId = new Map(nodes.map(node => [node.id, node]));
for (const node of nodes) {
  let parent = node.parent;
  const seen = new Set([node.id]);
  for (const _step of nodes) {
    if (!parent) break;
    if (!byId.has(parent) || seen.has(parent)) return false;
    seen.add(parent);
    parent = byId.get(parent).parent;
  }
}
return true;
}
