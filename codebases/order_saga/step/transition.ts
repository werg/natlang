import type { Event, Kind, Order, Command, State } from "../types.js";
export default function transition(acc: State, item: Event, kind: Kind): State {
const orders = Object.assign(Object.create(null), acc.orders);
const old = orders[item.order]; let state = old; let operations = [];
if (!old && kind === "start") { state = "reserved"; operations = ["reserve"]; }
else if (old === "reserved" && kind === "paid") { state = "paid"; operations = ["ship"]; }
else if (old === "paid" && kind === "sent") state = "done";
else if ((kind === "cancel" || kind === "failed") && (old === "reserved" || old === "paid")) {
  state = "cancelled"; operations = old === "paid" ? ["refund", "release"] : ["release"];
}
if (state) orders[item.order] = state;
return {seen: [...acc.seen, item.id], orders,
  outbox: operations.map((operation, i) => ({key: item.id + ":" + i, order: item.order, operation}))};
}
