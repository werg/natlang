import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { decide, nl, type TargetContext } from '@natlang/node';
import route from './route.nl';
import type { Message, Plan, Reply } from './types.js';

/** Plan the reader's inbox: route every message, draft replies for the ones that need one. */
export async function plan(messages: Message[]): Promise<Plan[]> {
  // Independent judgments run together; each is one scored decision on the local model.
  const decisions = await Promise.all(messages.map(message => decide(route, message)));
  const draft = nl<Reply>`Draft a reply to message for its reader.`;
  return Promise.all(messages.map(async (message, i) => {
    const { value, confidence } = decisions[i]!;
    const planned: Plan = { from: message.from, subject: message.subject, route: value, confidence };
    if (value === 'reply') planned.reply = await draft(message);
    return planned;
  }));
}

/** `natlang run`: plan the messages given as JSON, or the bundled messages.json. */
export async function main(context: TargetContext): Promise<number> {
  const given = context.arguments.messages as Message[] | undefined;
  const messages = given ?? JSON.parse(readFileSync(join(context.package?.root ?? context.workspace, 'messages.json'), 'utf8')) as Message[];
  context.io.output.write(JSON.stringify(await context.runtime.run(() => plan(messages)), null, 2) + '\n');
  return 0;
}
