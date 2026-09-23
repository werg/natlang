import type { State, Node, Edge, Item, Rule, Object, Event } from "../types.js";
export default function check_state(state: State): boolean {
const keys = state.rules.map(r => JSON.stringify([r.subject, r.resource, r.action]));
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       state.rules.every(r => r.subject.trim() && r.resource.trim() && r.action.trim()) &&
       new Set(keys).size === keys.length;
}
