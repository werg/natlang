/**
 * A refund desk for several shops. TypeScript holds the data and every exact step; each judgment is a typed
 * natural-language function: eligibility (a pluggable rule or judgment), the verdict (a scored decision under the
 * shop's policy, escalated when unsure) and the reply (refined until it answers the customer).
 */
import { Context, decide, type PluggableSetting } from '@natlang/node';
import verdict from './verdict.nl';
import { eligibility } from './rules.js';
import { replyFor } from './reply.js';
import type { Outcome, Request, Shop, Verdict } from './types.js';

export type DeskOptions = {
  /** Each shop's written refund policy. */
  policies: Record<Shop, string>;
  /** Below this confidence the stronger model settles the request. */
  floor?: number;
  /** The model unsure verdicts go to: a model of the runtime, such as the profile's escalation profile. */
  stronger?: string;
  /** `crisp` (default), `nl` or `shadow`. */
  eligibility?: PluggableSetting;
};

/** Settle every request at once: each one is independent. */
export async function settle(requests: Request[], options: DeskOptions): Promise<Outcome[]> {
  const eligible = eligibility(options.eligibility);
  // One verdict function, one context per shop: the shop's policy is the `policy` its instructions name.
  const under = (shop: Shop) => verdict.in(Context.empty().with({ policy: options.policies[shop] }));
  const judge = async (request: Request): Promise<{ value: Verdict; confidence: number; escalated: boolean }> => {
    if (!await eligible(request)) return { value: 'decline', confidence: 1, escalated: false };
    const policy = under(request.order.shop);
    const { value, confidence } = await decide(policy, request);
    if (confidence >= (options.floor ?? 0.8)) return { value, confidence, escalated: false };
    // Unsure: the same function in the same context, on the stronger model.
    return { value: await policy.on({ model: options.stronger ?? 'teacher' })(request), confidence, escalated: true };
  };
  return Promise.all(requests.map(async request => {
    const { value, confidence, escalated } = await judge(request);
    return { id: request.id, verdict: value, confidence, escalated, reply: await replyFor(request, value) };
  }));
}
