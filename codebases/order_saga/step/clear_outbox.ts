import type { Event, Kind, Order, Command, State } from "../types.js";
export default function clear_outbox(state: State): State {
return {...state, outbox: []};
}
