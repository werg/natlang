import type { Event, Kind, Order, Command, State } from "../types.js";
export default function seen(acc: State, item: Event): boolean {
return acc.seen.includes(item.id);
}
