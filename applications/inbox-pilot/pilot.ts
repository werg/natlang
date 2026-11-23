/**
 * Inbox Pilot (WS6 Tier 2 flagship A4):
 * - Crisp `pluggable` fast-path archives routine newsletters and receipts with zero model calls.
 * - Standing prose rules are `decideAction.nl`'s `rules` argument and can be compiled/verified with `compileRule`.
 * - Every external subject/body is fenced with `untrusted(...)` before any model sees it.
 * - Asymmetric `escalate(decideAction, { to: 'teacher', floor: 0.85, risk: { escalate: 0.25 } })` routes risky or
 *   uncertain messages (including prompt-injection attempts) to the stronger model and prevents unauthorized actions.
 * - `ONCE_EFFECTS` idempotent mail service wrapper ensures each message executes at most one mutating effect.
 */
import { ONCE_EFFECTS, pluggable, untrusted, type PluggableSetting } from '@natlang/node';
import { compileRule, escalate, semanticMap } from '@natlang/node/std';
import decideAction from './decideAction.nl';
import draftReply from './draftReply.nl';
import type {
  InboxAction,
  InboxReport,
  IncomingMessage,
  MailEffectResult,
  MailService,
  PilotDecision,
  ReplyDraft,
  StandingRule,
} from './types.js';

export type PilotOptions = {
  /** Pluggable mode for routine newsletter/receipt fast-path (default `'crisp'`). */
  fastPathMode?: PluggableSetting;
  /** Stronger model name for asymmetric escalation (default `'teacher'`). */
  escalateTo?: string;
  /** Minimum local confidence before escalating (default `0.85`). */
  floor?: number;
  /** Asymmetric risk floor for the `'escalate'` class (default `0.25`). */
  escalateRiskFloor?: number;
  /** Optional custom compiler passed to `compileRule` when compiling standing rules with examples. */
  ruleCompiler?: (rule: string, problems?: string[]) => Promise<(item: IncomingMessage) => Promise<InboxAction | null>>;
  /** Maximum concurrent message triage calls in `processInbox` (default `6`). */
  concurrency?: number;
};

/** Idempotent mail service wrapper declaring ONCE_EFFECTS and enforcing at-most-one effect per messageId. */
export type OnceMailService = MailService & {
  readonly [ONCE_EFFECTS]: readonly ['archive', 'saveDraft', 'hold'];
  executedCount(messageId: string): number;
};

function readHeader(headers: Record<string, string>, name: string): string | undefined {
  const lower = name.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === lower) return value;
  }
  return undefined;
}

/**
 * Exact crisp detector for routine newsletters and automated receipts.
 * Uses only structured transport headers and explicit receipt markers; never trusts body prose to approve actions.
 */
export function isRoutineNewsletterOrReceipt(message: IncomingMessage): boolean {
  const headers = message.headers ?? {};
  const listUnsub = readHeader(headers, 'List-Unsubscribe');
  const listId = readHeader(headers, 'List-Id');
  const precedence = readHeader(headers, 'Precedence')?.toLowerCase() ?? '';
  const autoSubmitted = readHeader(headers, 'Auto-Submitted')?.toLowerCase() ?? '';
  const receiptId = readHeader(headers, 'X-Receipt-Id') ?? readHeader(headers, 'X-Order-Receipt');
  const category = readHeader(headers, 'X-Mail-Category')?.toLowerCase() ?? '';

  if (listUnsub && listUnsub.trim().length > 0) return true;
  if (listId && (precedence === 'bulk' || precedence === 'list')) return true;
  if (receiptId && receiptId.trim().length > 0 && autoSubmitted.startsWith('auto-generated')) return true;
  if (category === 'newsletter' || category === 'receipt') return true;
  return false;
}

/** Fence every untrusted field of an incoming message before crossing a model boundary. */
export function fenceMessage(message: IncomingMessage): IncomingMessage {
  return {
    id: message.id,
    sender: message.sender,
    subject: untrusted(String(message.subject), `mail:${message.id}:subject`),
    body: untrusted(String(message.body), `mail:${message.id}:body`),
    headers: { ...message.headers },
  };
}

/**
 * Wrap a MailService with `ONCE_EFFECTS` metadata and per-messageId idempotency so each message executes
 * at most one mutating mail effect across retries or duplicate deliveries.
 */
export function createOnceMailService(inner: MailService): OnceMailService {
  // The first effect per messageId is kept as a promise, so concurrent duplicate deliveries share one effect.
  const byMessage = new Map<string, Promise<MailEffectResult>>();
  const counts = new Map<string, number>();
  const runOnce = async (
    messageId: string,
    effect: () => Promise<MailEffectResult> | MailEffectResult,
  ): Promise<MailEffectResult> => {
    const existing = byMessage.get(messageId);
    if (existing) {
      return { ...(await existing), deduplicated: true };
    }
    counts.set(messageId, (counts.get(messageId) ?? 0) + 1);
    const pending = Promise.resolve().then(effect);
    byMessage.set(messageId, pending);
    return pending;
  };
  return {
    [ONCE_EFFECTS]: ['archive', 'saveDraft', 'hold'] as const,
    archive: (messageId: string) => runOnce(messageId, () => inner.archive(messageId)),
    saveDraft: (messageId: string, reply: ReplyDraft) =>
      runOnce(messageId, () => inner.saveDraft(messageId, reply)),
    hold: (messageId: string, reason: string) =>
      runOnce(messageId, () => inner.hold(messageId, reason)),
    executedCount: (messageId: string) => counts.get(messageId) ?? 0,
  };
}

/**
 * Compile standing prose rules that supply validation examples into reusable typed rule functions via `compileRule`.
 */
export async function compileStandingRules(
  rules: StandingRule[],
  compiler?: PilotOptions['ruleCompiler'],
): Promise<Array<{ rule: StandingRule; evaluate: (msg: IncomingMessage) => Promise<InboxAction | null> }>> {
  const compiled: Array<{ rule: StandingRule; evaluate: (msg: IncomingMessage) => Promise<InboxAction | null> }> = [];
  for (const rule of rules) {
    if (!rule.examples || rule.examples.length === 0) continue;
    const evaluate = await compileRule<IncomingMessage, InboxAction | null>(rule.guidance, {
      examples: rule.examples,
      ...(compiler ? { compiler } : {}),
    });
    compiled.push({ rule, evaluate });
  }
  return compiled;
}

/** Triage and apply at most one idempotent mail effect for a single incoming message. */
export async function triageMessage(
  rawMessage: IncomingMessage,
  rules: StandingRule[],
  service: MailService,
  options: PilotOptions = {},
  compiledRules: Array<{ rule: StandingRule; evaluate: (msg: IncomingMessage) => Promise<InboxAction | null> }> = [],
): Promise<PilotDecision> {
  const message = fenceMessage(rawMessage);
  const onceService = ONCE_EFFECTS in service ? (service as OnceMailService) : createOnceMailService(service);
  const safeDecide = escalate(decideAction, {
    to: options.escalateTo ?? 'teacher',
    floor: options.floor ?? 0.85,
    risk: { escalate: options.escalateRiskFloor ?? 0.25 },
  });
  // One semantic triage decision per message, shared by the fast path's nl side and the main decision below.
  let pendingDecision: ReturnType<typeof safeDecide> | undefined;
  const decideOnce = () => (pendingDecision ??= safeDecide(message, rules));
  // Pluggable fast path: is this routine bulk mail to archive? crisp reads transport headers (0 model calls); nl asks
  // the triage decision; shadow runs both, records agreement and serves crisp.
  const fastPath = pluggable(
    {
      crisp: (msg: IncomingMessage): boolean => isRoutineNewsletterOrReceipt(msg),
      nl: async (): Promise<boolean> => (await decideOnce()).value === 'archive',
    },
    options.fastPathMode,
    { default: 'crisp', serve: 'crisp', name: 'inbox-routine-fast-path' },
  );
  const fastMode = options.fastPathMode ?? 'crisp';
  if (fastMode !== 'nl' && await fastPath(message)) {
    const effect = await onceService.archive(message.id);
    return {
      messageId: message.id,
      action: 'archive',
      source: 'crisp',
      confidence: 1,
      escalated: false,
      effect,
    };
  }
  // Evaluate compiled standing rules if available (only for non-escalate matches; escalate still goes to safe hold).
  for (const entry of compiledRules) {
    const ruleAction = await entry.evaluate(message);
    if (ruleAction && ruleAction !== 'escalate') {
      if (ruleAction === 'archive') {
        const effect = await onceService.archive(message.id);
        return { messageId: message.id, action: 'archive', source: 'compiled-rule', confidence: 1, escalated: false, effect };
      }
      if (ruleAction === 'hold') {
        const effect = await onceService.hold(message.id, `Held by standing rule "${entry.rule.name}"`);
        return { messageId: message.id, action: 'hold', source: 'compiled-rule', confidence: 1, escalated: false, effect };
      }
      if (ruleAction === 'draft-reply') {
        const draft = await draftReply(message, entry.rule.guidance);
        const effect = await onceService.saveDraft(message.id, draft);
        return {
          messageId: message.id,
          action: 'draft-reply',
          source: 'compiled-rule',
          confidence: 1,
          escalated: false,
          draft: String(draft),
          effect,
        };
      }
    }
  }

  const decision = await decideOnce();
  const guidanceText = rules.map(r => `${r.name}: ${r.guidance}`).join('\n');

  if (decision.value === 'archive') {
    const effect = await onceService.archive(message.id);
    return {
      messageId: message.id,
      action: 'archive',
      source: 'nl',
      confidence: decision.confidence,
      escalated: decision.escalated,
      ...(decision.reason ? { escalationReason: decision.reason } : {}),
      effect,
    };
  }

  if (decision.value === 'draft-reply') {
    const draft = await draftReply(message, guidanceText || 'Reply politely without making commitments.');
    const effect = await onceService.saveDraft(message.id, draft);
    return {
      messageId: message.id,
      action: 'draft-reply',
      source: 'nl',
      confidence: decision.confidence,
      escalated: decision.escalated,
      ...(decision.reason ? { escalationReason: decision.reason } : {}),
      draft: String(draft),
      effect,
    };
  }

  // Both 'hold' and 'escalate' execute only a non-destructive hold so unauthorized actions are blocked.
  const holdReason = decision.value === 'escalate'
    ? `Escalated for human review${decision.reason ? ` (${decision.reason})` : ''}`
    : 'Held for human review under standing rules';
  const effect = await onceService.hold(message.id, holdReason);
  return {
    messageId: message.id,
    action: decision.value,
    source: 'nl',
    confidence: decision.confidence,
    escalated: decision.escalated,
    ...(decision.reason ? { escalationReason: decision.reason } : {}),
    effect,
  };
}

/** Process a batch of incoming messages concurrently through an idempotent ONCE_EFFECTS mail service. */
export async function processInbox(
  messages: IncomingMessage[],
  rules: StandingRule[],
  service: MailService,
  options: PilotOptions = {},
): Promise<InboxReport> {
  const onceService = createOnceMailService(service);
  const compiledRules = options.ruleCompiler ? await compileStandingRules(rules, options.ruleCompiler) : [];
  const decisions = await semanticMap(
    messages,
    message => triageMessage(message, rules, onceService, options, compiledRules),
    { concurrency: options.concurrency ?? 6 },
  );
  return {
    processed: decisions.length,
    crispServed: decisions.filter(d => d.source === 'crisp').length,
    escalatedCount: decisions.filter(d => d.escalated || d.action === 'escalate').length,
    decisions,
  };
}
