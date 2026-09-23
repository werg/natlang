export default function check_state(state: State): boolean {
const ids = state.items.map(item => item.id);
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && new Set(ids).size === ids.length;
}
