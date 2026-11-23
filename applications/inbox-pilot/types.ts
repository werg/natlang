/** One external incoming message; subject and body are untrusted external text. */
export type IncomingMessage = {
  id: string;
  sender: string;
  subject: Untrusted<string>;
  body: Untrusted<string>;
  headers: Record<string, string>;
};

/** Standing prose rule governing how matching messages should be handled. */
export type StandingRule = {
  id: string;
  name: string;
  guidance: string;
  examples?: {
    input: IncomingMessage;
    expected: InboxAction;
    name?: string;
  }[];
};

/** Finite triage action space for inbox messages. */
export type InboxAction = 'archive' | 'draft-reply' | 'hold' | 'escalate';

/** A drafted reply refined to follow the active standing rule without making unapproved commitments. */
export type ReplyDraft = Is<
  string,
  'a concise, polite reply that follows the standing rule and makes no unapproved commitment'
>;

/** Structured receipt for a single mutating mail effect. */
export type MailEffectResult = {
  messageId: string;
  effect: 'archive' | 'saveDraft' | 'hold';
  ok: boolean;
  draft?: string;
  reason?: string;
  deduplicated?: boolean;
};

/** External mail service boundary with idempotent ONCE_EFFECTS actions keyed by messageId. */
export type MailService = {
  archive(messageId: string): Promise<MailEffectResult> | MailEffectResult;
  saveDraft(messageId: string, reply: ReplyDraft): Promise<MailEffectResult> | MailEffectResult;
  hold(messageId: string, reason: string): Promise<MailEffectResult> | MailEffectResult;
};

/** Outcome of processing one incoming message. */
export type PilotDecision = {
  messageId: string;
  action: InboxAction;
  source: 'crisp' | 'compiled-rule' | 'nl';
  confidence: number;
  escalated: boolean;
  escalationReason?: string;
  draft?: string;
  effect: MailEffectResult;
};

/** Aggregate report for a batch of processed messages. */
export type InboxReport = {
  processed: number;
  crispServed: number;
  escalatedCount: number;
  decisions: PilotDecision[];
};
