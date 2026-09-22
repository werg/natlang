/*---
description: Update stock, coins and the ledger for an action.
args:
  acc: Shop
  action: Action
  customer: string
returns: Shop
---*/
const a = args.action
if (!a.code.startsWith("sell")) return { ...args.acc, ledger: args.acc.ledger.concat([`${args.customer}: ${a.code}`]).slice(-100) }
return { ...args.acc, stock: { ...args.acc.stock, [a.good]: args.acc.stock[a.good] - a.qty }, coins: args.acc.coins + a.qty * a.price,
         ledger: args.acc.ledger.concat([`${args.customer}: ${a.code} ${a.qty} ${a.good} @ ${a.price}`]).slice(-100) }
