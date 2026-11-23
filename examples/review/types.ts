export type Assessment = { verdict: "supported" | "contradicted" | "uncertain", reason: string };
export type Tally = { assessments: Assessment[], supported: number, contradicted: number, uncertain: number };
export type Conclusion = Is<string, "one sentence stating whether the criterion holds overall and naming any contradiction">;
export type Report = Tally & { conclusion: Conclusion };
