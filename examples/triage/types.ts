export type Label = "billing" | "technical" | "spam";
/** What the report says about the urgent tickets. */
export type Summary = Is<string, "one paragraph that names the most severe problem first">;
export type Report = { urgent: number, by_label: Record<string, number>, summary: string };
