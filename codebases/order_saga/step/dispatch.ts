import type { Event, Kind, Order, Command, State } from "../types.js";
import { effects as fx } from "natlang:runtime";

export default function dispatch(commands: Command[]): number {
for (const command of commands) fx.queue.send(command);
return commands.length;
}
