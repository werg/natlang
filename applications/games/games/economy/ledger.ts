/**
 * Exact bookkeeping over merchants: the entries of a traded outcome, applying entries to balances, the totals
 * of an economy. Plain arithmetic over named data; the settlement stages and the commit share these definitions.
 */
import type { Entry, Merchant, Totals, TradeOutcome } from '../../types.js';

/** The two entries of a traded outcome (the goods from seller to buyer, then the money from buyer to seller); none otherwise. */
export function entriesOf(outcome: TradeOutcome): Entry[] {
  if (outcome.status !== 'traded') return [];
  return [
    { kind: 'good', from: outcome.seller!, to: outcome.actor, good: outcome.good!, amount: outcome.quantity! },
    { kind: 'cash', from: outcome.actor, to: outcome.seller!, amount: outcome.total! },
  ];
}

/** The merchants after the entries, in order. Merchants are copied; an entry between unknown ids is skipped. */
export function post(merchants: Merchant[], entries: Entry[]): Merchant[] {
  const next: Merchant[] = structuredClone(merchants);
  for (const entry of entries) {
    const from = next.find(row => row.id === entry.from), to = next.find(row => row.id === entry.to);
    if (!from || !to) continue;
    if (entry.kind === 'cash') { from.cash -= entry.amount; to.cash += entry.amount; }
    else { from.goods[entry.good!] = (from.goods[entry.good!] ?? 0) - entry.amount; to.goods[entry.good!] = (to.goods[entry.good!] ?? 0) + entry.amount; }
  }
  return next;
}

/** The first balance below zero as a sentence, or an empty string when every cash and goods balance is at least 0. */
export function lowest(merchants: Merchant[]): string {
  for (const row of merchants) {
    if (row.cash < 0) return `${row.id} would hold negative cash`;
    for (const [good, quantity] of Object.entries(row.goods)) if (quantity < 0) return `${row.id} would hold negative ${good}`;
  }
  return '';
}

/** The money and each good summed over the merchants, goods sorted by name. */
export function totalsOf(merchants: Merchant[]): Totals {
  const goods: Record<string, number> = {};
  for (const row of merchants) for (const [name, quantity] of Object.entries(row.goods)) goods[name] = (goods[name] ?? 0) + quantity;
  return { cash: merchants.reduce((sum, row) => sum + row.cash, 0),
    goods: Object.fromEntries(Object.entries(goods).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) };
}
