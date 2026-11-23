import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { decide, nl, untrusted, type TargetContext } from '@natlang/node';
import route from './route.nl';
import type { Message, Plan, Reply } from './types.js';

/** Plan the reader's inbox: route every message, escalate unsure ones, and draft replies where needed. */
export async function plan(messages: Message[]): Promise<Plan[]> {
  // Independent judgments run together; each is a calibrated decision on the local model.
  const decisions = await Promise.all(messages.map(message => decide(route, message)));
  const draft = nl<Reply>`Draft a reply to message for its reader.`;
  return Promise.all(messages.map(async (message, i) => {
    const { value, confidence } = decisions[i]!;
    // Unsure messages escalate to the stronger model; high-confidence ones settle locally.
    const chosen = confidence >= 0.8 ? value : await route.on({ model: 'teacher' })(message);
    const planned: Plan = { from: message.from, subject: message.subject, route: chosen, confidence };
    if (chosen === 'reply') planned.reply = await draft(message);
    return planned;
  }));
}

/** `natlang run`: plan the messages given as JSON, or the bundled messages.json. */
export async function main(context: TargetContext): Promise<number> {
  const given = context.arguments.messages as Message[] | undefined;
  const raw = given ?? JSON.parse(readFileSync(join(context.package?.root ?? context.workspace, 'messages.json'), 'utf8')) as Message[];
  const messages = raw.map(message => ({ ...message, body: untrusted(message.body, `message from ${message.from}`) }));
  context.io.output.write(JSON.stringify(await context.runtime.run(() => plan(messages)), null, 2) + '\n');
  return 0;
}
