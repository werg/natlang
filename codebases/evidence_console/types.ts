export type Claim = { text: string, span_id: string, revision: string, quote: string };
export type EvidenceAnswer = { status: string, answer: string, claims: Claim[], gaps: string[], collection_revision: string, detail: string };
export type ConsoleEvent = { id: string, kind: string, value: string };
export type ConsoleState = { questions: string[], answers: EvidenceAnswer[], status: string };
export type ViewBlock = { kind: string, text?: string, tone?: string, items?: string[], ordered?: boolean, columns?: string[], rows?: string[][] };
export type TerminalView = { title?: string, subtitle?: string, blocks: ViewBlock[], prompt?: string, busy?: boolean, help?: string[] };
