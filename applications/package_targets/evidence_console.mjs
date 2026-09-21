import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { EvidenceCollection } from '../evidence_atlas.mjs';
import { flag, terminalExecutable } from './terminal_helpers.mjs';

export function createTarget(context) {
  const path = flag(context.args, '--documents');
  if (!path) throw new Error('pass -- --documents FILE containing [{id,text}, ...]');
  const documents = JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8'));
  if (!Array.isArray(documents)) throw new Error('documents file must contain an array');
  const evidence = new EvidenceCollection(documents);
  return terminalExecutable(context, { hostObject: { evidence, drainEvents: () => evidence.drainEvents() },
    initialState: () => ({ questions: [], answers: [], status: 'idle' }),
    event: (value, id) => ({ id, kind: 'question', value }) });
}
