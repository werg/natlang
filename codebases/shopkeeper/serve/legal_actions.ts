export default function legal_actions(acc: Shop, intent: Intent): Action[] {
const i = intent, out = [{ code: "decline", good: "", qty: 0, price: 0 }]
const price = acc.prices[i.good], have = acc.stock[i.good] || 0
if (price === undefined) return out.concat([{ code: "chat", good: "", qty: 0, price: 0 }])
out.push({ code: "quote", good: i.good, qty: 0, price })
const qty = Math.max(1, Math.min(i.qty || 1, have))
if (have > 0 && (i.kind === "buy" || i.kind === "haggle")) {
  out.push({ code: "sell_list_price", good: i.good, qty, price })
  const floor = Math.ceil(price * 0.8)                       // never below 80% of the list price
  if (i.kind === "haggle" && i.offer >= floor) out.push({ code: "sell_at_offer", good: i.good, qty, price: Math.min(i.offer, price) })
  if (i.kind === "haggle" && i.offer < floor) out.push({ code: "counter_offer", good: i.good, qty, price: floor })
}
if (have === 0) out.push({ code: "out_of_stock", good: i.good, qty: 0, price })
return out
}
