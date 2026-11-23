import { iterateOn, nl } from '@natlang/node';
import type { Reply, Request, Verdict } from './types.js';

/** Write the customer's reply, then revise it until it answers everything they asked. */
export async function replyFor(request: Request, verdict: Verdict): Promise<Reply> {
  const first: Reply = await nl`Write the reply to request that tells the customer the verdict.`();
  const revise = nl<(reply: Reply) => Reply>`Revise reply so that it also answers every question the customer asks in request. Keep the verdict.`;
  return iterateOn(revise, first).until(nl<(reply: Reply) => boolean>`reply answers every question the customer asks in request.`);
}
