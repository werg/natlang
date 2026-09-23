import type { Offer, MarketObservation, TradeIntent, TradeReceipt } from "../types.js";
import { host } from "natlang:runtime";

export default function submit(actor: string, tick: number, intent: TradeIntent): TradeReceipt {
return host.economy.submit(actor, tick, intent);
}
