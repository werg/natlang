import type { Untrusted } from '@natlang/node';

export type Urgency = 'low' | 'normal' | 'urgent';
/** One message of a conversation, as stored. */
export type Message = { from: 'customer' | 'agent', author: string, text: string, at: number };
/** A message as the natural-language stages read it: the text comes from a customer or an agent outside the program, so models see it as quoted data. */
export type Line = { from: 'customer' | 'agent', text: Untrusted<string> };
/** What the stages made of the open request. topic names the product area; summary is one sentence an agent can act on. */
export type Triage = { urgency: Urgency, topic: string, summary: string };
/** Whom an escalation notifies. */
export type Notify = 'supervisor' | 'on-call';
/**
 * What an escalation asks of the outside world, as data. note goes to whoever is notified. holding_reply is text for an
 * agent to send to the customer; the desk never sends it.
 */
export type EscalationPlan = { notify: Notify, note: string, holding_reply: string | null };
/** What the escalation plan reads about a ticket that went past its deadline. */
export type EscalationFacts = {
  urgency: Urgency, topic: string, summary: string,
  /** Details the agent still needs from the customer (empty when none are known). */
  missing: string[],
  /** Whole minutes the customer has waited since they started waiting. */
  waited_minutes: number,
  /** Whole minutes the ticket was allowed to wait. */
  allowed_minutes: number,
  /** How many customer messages were written after the deadline passed. */
  late_messages: number,
};
/** A ticket the inbox ranking reads. due is a time in milliseconds, or null when nobody is waiting. */
export type RankedTicket = { id: string, urgency: Urgency | null, topic: string, summary: string, escalated: boolean, due: number | null };

export type Ticket = {
  id: string,
  customer: string,
  messages: Message[],
  status: 'open' | 'answered' | 'closed',
  triage: Triage | null,
  /** How many messages the triage read: a triage of more messages replaces one of fewer, never the reverse. */
  triaged: number,
  /** A suggested reply for the agent, for the conversation as it stands (dropped when it moves on). */
  draft: string | null,
  /** The details the draft asks the customer for (the agent's checklist); empty when the agent can answer now. */
  missing: string[],
  /** How long the customer may wait, in milliseconds, under the latest triage; null before the first triage. */
  allowedMs: number | null,
  /** When an agent must have answered the customer; null while nobody is waiting. */
  due: number | null,
  escalated: boolean,
  /** The plan made when the ticket was escalated; null until then. */
  escalation: EscalationPlan | null,
};

// ---------------------------------------------------------------- refined results
// Each predicate has a crisp checker in refinements.ts. A property that needs more than the value (the offered ids of a
// ranking, the messages of a ticket) is checked by the desk in index.ts instead.

/** Two to four words naming a product area. */
export type CheckedTopic = Is<string, "two to four words naming a product area">;
/** One sentence that says what is wrong or wanted and what the customer already tried. */
export type CheckedSummary = Is<string, "one sentence that states what is wrong or wanted and what the customer already tried">;
/** A reply for the customer. */
export type CheckedDraft = Is<string, "two to five sentences of plain text addressed to the customer">;
/** A number of minutes. */
export type CheckedMinutes = Is<number, "a positive whole number of minutes">;
/** What summarize returns. */
export type Summary = { topic: CheckedTopic, summary: CheckedSummary };
