export default function apply_action(acc: Shop, action: Action, customer: string): Shop {
const a = action
if (!a.code.startsWith("sell")) return { ...acc, ledger: acc.ledger.concat([`${customer}: ${a.code}`]).slice(-100) }
return { ...acc, stock: { ...acc.stock, [a.good]: acc.stock[a.good] - a.qty }, coins: acc.coins + a.qty * a.price,
         ledger: acc.ledger.concat([`${customer}: ${a.code} ${a.qty} ${a.good} @ ${a.price}`]).slice(-100) }
}
