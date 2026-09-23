import type { Receipt, State, Step, Decision, UiEvent, View } from "../types.js";

export default function finish(step: Step): State {
return step.state;
}
