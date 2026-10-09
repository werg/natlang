import type { MarketObservation, EconomyState } from "../../types.js";
/** What one merchant knows: its own cash and goods, and the public offers of the others. The information boundary. */
export default function observe(state: EconomyState, actor: string): MarketObservation {
  const mine = state.merchants.find(row => row.id === actor);
  if (!mine) throw new Error('unknown actor');
  return { actor, tick: state.tick, cash: mine.cash, goods: structuredClone(mine.goods),
    offers: state.merchants.filter(row => row.id !== actor)
      .flatMap(row => Object.entries(row.offers).map(([good, price]) => ({ seller: row.id, good, price }))) };
}
