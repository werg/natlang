export type Event = { kind: string, id: string, request_id: string, job_id: string, text: string, status: string, detail: string };
export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };
export type Recipe = { id: string, description: string };
export type Job = { id: string, request_id: string, status: string, detail: string };
export type Outcome = { request_id: string, job_id: string, status: string, detail: string };
export type Session = { revision: number, active_request: string, active_job: string, status: string, messages: string[], history: Outcome[] };
export type ViewBlock = { kind: string, text?: string, tone?: string, items?: string[], ordered?: boolean, columns?: string[], rows?: string[][] };
export type TerminalView = { title?: string, subtitle?: string, blocks: ViewBlock[], prompt?: string, busy?: boolean, help?: string[] };
