export type Mark = "X" | "O";
export type Cell = "X" | "O" | "empty";
export type Verdict = { legal: boolean, reason: string, wins: boolean, board: string };
