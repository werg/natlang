/*---
args:
  acc: State
  item: Event
  kind: Kind
returns: State
---*/
const orders = Object.assign(Object.create(null), args.acc.orders);
const old = orders[args.item.order]; let state = old; let operations = [];
if (!old && args.kind === "start") { state = "reserved"; operations = ["reserve"]; }
else if (old === "reserved" && args.kind === "paid") { state = "paid"; operations = ["ship"]; }
else if (old === "paid" && args.kind === "sent") state = "done";
else if ((args.kind === "cancel" || args.kind === "failed") && (old === "reserved" || old === "paid")) {
  state = "cancelled"; operations = old === "paid" ? ["refund", "release"] : ["release"];
}
if (state) orders[args.item.order] = state;
return {seen: [...args.acc.seen, args.item.id], orders,
  outbox: operations.map((operation, i) => ({key: args.item.id + ":" + i, order: args.item.order, operation}))};
