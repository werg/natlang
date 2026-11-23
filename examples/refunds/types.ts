/** The order a request is about, as the shop's records hold it. */
export type Order = { id: string; shop: Shop; total: number; days_since_delivery: number; opened: boolean };
/** The shops this desk serves; each has its own written refund policy (policies/<shop>.md). */
export type Shop = 'outdoor' | 'books';
/** A refund request as it arrives. The customer's text is outside data: the runtime fences it and never follows it. */
export type Request = { id: string; order: Order; text: Untrusted<string> };
/** How a request is settled. */
export type Verdict = 'refund' | 'partial' | 'decline';
/** What the customer is told. */
export type Reply = Is<string, 'two to four polite sentences that state the decision and promise nothing beyond it'>;
/** One settled request: the verdict, how sure the local model was, and whether a stronger model was asked. */
export type Outcome = { id: string; verdict: Verdict; confidence: number; escalated: boolean; reply: Reply };
/** Settled requests counted by verdict. */
export type Ledger = { refund: number; partial: number; decline: number };
