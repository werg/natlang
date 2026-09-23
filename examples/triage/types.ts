export type Label = "billing" | "technical" | "spam";
export type Report = { urgent: number, by_label: Record<string, number>, summary: string };
