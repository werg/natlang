import type { Entry, Merchant, Settlement, Submission, TradeOutcome } from "../../../types.js";
/** The crisp settlement: each intent against the running balances in the order given. The reference of settle/policy.nl. */
export default function reference(merchants: Merchant[], ordered: Submission[]): Settlement {
  const running: Merchant[] = structuredClone(merchants);
  const outcomes: TradeOutcome[] = [], entries: Entry[] = [];
  for (const { actor, intent } of ordered) {
    if (intent.kind === 'pass') { outcomes.push({ actor, status: 'pass' }); continue; }
    const buyer = running.find(row => row.id === actor)!, seller = running.find(row => row.id === intent.seller);
    const good = intent.good!, quantity = intent.quantity!;
    const price = seller?.offers[good];
    const reject = (reason: string) => outcomes.push({ actor, status: 'rejected', reason });
    if (!seller || seller.id === actor) { reject('seller is not another merchant'); continue; }
    if (!Number.isSafeInteger(price)) { reject('seller does not offer the good'); continue; }
    if (!Number.isSafeInteger(quantity * price!)) { reject('total is not a safe integer'); continue; }
    if ((seller.goods[good] ?? 0) < quantity) { reject('seller lacks the stock'); continue; }
    if (buyer.cash < quantity * price!) { reject('buyer lacks the cash'); continue; }
    const total = quantity * price!;
    seller.goods[good]! -= quantity; buyer.goods[good] = (buyer.goods[good] ?? 0) + quantity;
    buyer.cash -= total; seller.cash += total;
    outcomes.push({ actor, status: 'traded', seller: seller.id, good, quantity, total });
    entries.push({ kind: 'good', from: seller.id, to: actor, good, amount: quantity }, { kind: 'cash', from: actor, to: seller.id, amount: total });
  }
  return { outcomes, entries };
}
