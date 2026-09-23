import type { State, Node, Edge, Item, Rule, Object, Event } from "../types.js";
export default function check_state(state: State): boolean {
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       Number.isSafeInteger(state.value) && state.unit.trim().length > 0;
}
