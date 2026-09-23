import type { Message, Shop, Intent, Action } from "../types.js";
export default function goods_of(acc: Shop): string[] {
return Object.keys(acc.prices)
}
