export type Offer = { seller: string, good: string, price: number };
export type MarketObservation = { actor: string, tick: number, cash: number,
  goods: Record<string, number>, offers: Offer[] };
export type TradeIntent = { kind: string, seller?: string, good?: string,
  quantity?: number };
export type TradeReceipt = { status: string, tick: number, actor: string };
