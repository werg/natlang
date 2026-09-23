import type { Message, Shop, Intent, Action } from "../types.js";
import { effects as fx } from "natlang:runtime";

export default function emit_reply(to: string, line: string, action: Action): boolean {
fx.out.emit({ to: to, line: line, action: action.code })
return true
}
