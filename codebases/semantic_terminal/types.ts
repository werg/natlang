export type Event = { kind: Text, id: Text, request_id: Text, job_id: Text, text: Text, status: Text, detail: Text };
export type File = { kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num };
export type Recipe = { id: Text, description: Text };
export type Job = { id: Text, request_id: Text, status: Text, detail: Text };
export type Outcome = { request_id: Text, job_id: Text, status: Text, detail: Text };
export type Session = { revision: Num, active_request: Text, active_job: Text, status: Text, messages: Text[], history: Outcome[] };
export type ViewBlock = { kind: Text, text?: Text, tone?: Text, items?: Text[], ordered?: Bool, columns?: Text[], rows?: Text[][] };
export type TerminalView = { title?: Text, subtitle?: Text, blocks: ViewBlock[], prompt?: Text, busy?: Bool, help?: Text[] };
