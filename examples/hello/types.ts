/** One incoming message. The body is outside text: the runtime fences it and never follows instructions in it. */
export type Message = { from: string; subject: string; body: Untrusted<string> };
/** What the message needs from its reader. */
export type Route = 'reply' | 'fyi' | 'ignore';
/** A drafted reply: short, polite, and promising nothing the message does not ask for. */
export type Reply = Is<string, 'a polite reply of at most three sentences that makes no commitment the message does not ask for'>;
export type Plan = { from: string; subject: string; route: Route; confidence: number; reply?: Reply };
