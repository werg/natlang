export type Answer = { verdict: "yes" | "unknown", derived: string[] };
export type State = { known: string[], derived: string[], grew: boolean };
