import type { Offer, MarketObservation, TradeIntent, TradeReceipt } from "../types.js";
import { host } from "natlang:runtime";

export default function observe(actor: string): MarketObservation {
return host.economy.observe(actor);
}
