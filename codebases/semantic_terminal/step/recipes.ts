import type { Event, File, Recipe, Job, Outcome, Session, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function recipes(): Recipe[] {
return host.terminal.catalog();
}
