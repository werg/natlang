export type Offer = { seller: Text, good: Text, price: Num };
export type MarketObservation = { actor: Text, tick: Num, cash: Num,
  goods: Dict<Num>, offers: Offer[] };
export type TradeIntent = { kind: Text, seller?: Text, good?: Text,
  quantity?: Num };
export type TradeReceipt = { status: Text, tick: Num, actor: Text };
