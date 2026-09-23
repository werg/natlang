export default function check_state(state: State): boolean {
const ids = state.nodes.map(node => node.id);
const known = new Set(ids);
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && known.size === ids.length &&
       state.edges.every(edge => known.has(edge.from) && known.has(edge.to) && edge.relation.trim());
}
