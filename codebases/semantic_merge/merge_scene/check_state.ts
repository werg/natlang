import type { State, Node, Edge, Item, Rule, Object, Event } from "../types.js";
export default function check_state(state: State): boolean {
const ids = state.objects.map(object => object.id);
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && new Set(ids).size === ids.length &&
       state.objects.every(object => Number.isSafeInteger(object.x) && Number.isSafeInteger(object.y));
}
