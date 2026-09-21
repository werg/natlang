import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { NotebookWorkspace } from '../notebook.mjs';
import { flag, terminalExecutable } from './terminal_helpers.mjs';

export function createTarget(context) {
  const path = flag(context.args, '--notebook');
  if (!path) throw new Error('pass -- --notebook FILE containing {cells,tables}');
  const config = JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8'));
  const Environment = context.runtime.TypeScriptEnvironment;
  const notebook = new NotebookWorkspace(config.cells, config.tables ?? {},
    { environment: new Environment({ mode: 'fresh' }) });
  return terminalExecutable(context, { hostObject: { notebook, drainEvents: () => notebook.drainEvents() },
    initialState: () => ({ requests: [], runs: [], status: 'idle' }),
    event: (value, id) => ({ id, kind: 'request', value }), close: () => notebook.close() });
}
