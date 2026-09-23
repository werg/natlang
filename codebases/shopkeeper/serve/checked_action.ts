import type { Message, Shop, Intent, Action } from "../types.js";
export default function checked_action(wish: string, legal: Action[]): Action {
return legal.find(a => a.code === wish.trim()) || legal.find(a => a.code === "decline")
}
