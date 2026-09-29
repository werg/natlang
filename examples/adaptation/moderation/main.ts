import moderate from './moderate.nl';
export const review = moderate;
// Exercise the same compositional helpers independently so local scoring does not
// infer their behavior from the root function's answer.
export const inspectRisk = moderate.risk;
export const inspectDecision = moderate.decision;
