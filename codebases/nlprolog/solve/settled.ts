import type { Answer, State } from "../types.js";
export default function settled(state: State): boolean {
return !state.grew
}
