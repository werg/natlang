import type { State, Node, Edge, Item, Rule, Object, Event } from "../types.js";
export default function check_state(state: State): boolean {
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       state.members.every(member => member.trim().length > 0) &&
       new Set(state.members).size === state.members.length;
}
