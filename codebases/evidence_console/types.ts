export type Claim = { text: Text, span_id: Text, revision: Text, quote: Text };
export type EvidenceAnswer = { status: Text, answer: Text, claims: Claim[], gaps: Text[], collection_revision: Text, detail: Text };
export type ConsoleEvent = { id: Text, kind: Text, value: Text };
export type File = { kind: Text, text?: Text, bytes: Num };
export type ConsoleState = { questions: Text[], answers: EvidenceAnswer[], status: Text };
export type ViewBlock = { kind: Text, text?: Text, tone?: Text, items?: Text[], ordered?: Bool, columns?: Text[], rows?: Text[][] };
export type TerminalView = { title?: Text, subtitle?: Text, blocks: ViewBlock[], prompt?: Text, busy?: Bool, help?: Text[] };
