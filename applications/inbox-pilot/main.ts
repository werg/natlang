/** `natlang run applications/inbox-pilot -- [INPUT.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { createOnceMailService, processInbox, type PilotOptions } from './pilot.js';
import type { IncomingMessage, MailEffectResult, MailService, StandingRule } from './types.js';

type CliPayload = {
  messages: IncomingMessage[];
  rules?: StandingRule[];
  options?: Pick<PilotOptions, 'fastPathMode' | 'escalateTo' | 'floor' | 'escalateRiskFloor' | 'compileRules' | 'concurrency'>;
};

function createRecordingMailService(): MailService {
  return {
    archive: async (messageId): Promise<MailEffectResult> => ({ ok: true, effect: 'archive', messageId }),
    saveDraft: async (messageId, reply): Promise<MailEffectResult> => ({ ok: true, effect: 'saveDraft', messageId, draft: reply }),
    hold: async (messageId, reason): Promise<MailEffectResult> => ({ ok: true, effect: 'hold', messageId, reason }),
  };
}

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;
  const service = createOnceMailService(createRecordingMailService());
  const report = await context.runtime.run(() =>
    processInbox(payload.messages ?? [], payload.rules ?? [], service, payload.options ?? {}),
  );
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return 0;
}
