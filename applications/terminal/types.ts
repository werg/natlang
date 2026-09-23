export type Recipe = { id: string, description: string };
export type TerminalEvent = { kind: "request" | "complete" | "cancel" | "recover", id: string, request_id: string, job_id: string,
  text: string, status: string, detail: string };
