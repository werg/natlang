import type { Cell, State, Step, Decision, UiEvent, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function apply(state: State, event: UiEvent, decision: Decision): Promise<Step> {
return await host.studio.apply(state, event, decision);
}
