export type Assessment = { verdict: "supported" | "contradicted" | "uncertain", reason: string };
export type Report = { assessments: Assessment[], supported: number, contradicted: number, uncertain: number };
