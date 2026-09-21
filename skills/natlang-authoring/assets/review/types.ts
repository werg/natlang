export type Assessment = { verdict: "supported" | "contradicted" | "uncertain", reason: Text };
export type Report = { assessments: Assessment[], supported: Num, contradicted: Num, uncertain: Num };
