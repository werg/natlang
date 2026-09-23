/** Notebook Console: ask for results; natlang picks the goal cell, runs its graph, and explains it. */
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { EventLoop, TerminalSessionStore, openFolder, runTerminalShell, type TargetContext, type TerminalView } from '@natlang/node';
import { NotebookWorkspace, STARTER_NOTEBOOK, answerRequest, type NotebookConfig, type NotebookRun } from './index.js';

export type ConsoleState = { requests: string[], runs: NotebookRun[], status: string };
type Request = { id: string, kind: 'request', value: string };

export function view(state: ConsoleState): TerminalView {
  const run = state.runs.at(-1), request = state.requests.at(-1);
  const blocks: TerminalView['blocks'] = run ? [
    { kind: 'text', text: `Request: ${request}` },
    { kind: 'status', text: `${run.status}: ${run.detail}`, tone: run.status === 'done' ? 'good' : 'warn' },
    { kind: 'text', text: run.answer || 'No answer was produced.' },
    { kind: 'table', columns: ['Cell', 'Status', 'Revision', 'Sample'],
      rows: run.results.map(result => [result.id, result.status, String(result.revision), result.sample || result.detail]) },
    ...(run.blocked.length ? [{ kind: 'list' as const, items: run.blocked.map(id => `Blocked: ${id}`) }] : []),
  ] : [{ kind: 'text', text: 'A starter notebook is ready. Ask for a result, inspect /cells, or import your own notebook with /load FILE.', tone: 'muted' },
    { kind: 'list', items: ['Try: Which region has the largest total, and what is that total?', 'Use /help to discover setup and navigation commands.'] }];
  return { title: 'Natlang Notebook Console', subtitle: `${state.runs.length} runs`, blocks,
    prompt: 'notebook> ', help: ['/help commands', '/cells catalog', '/load FILE import notebook', '/quit exit'] };
}

export async function main(context: TargetContext): Promise<number> {
  const at = context.args.indexOf('--notebook');
  const config: NotebookConfig = at >= 0 ? JSON.parse(readFileSync(resolve(context.workspace, context.args[at + 1]!), 'utf8')) : STARTER_NOTEBOOK;
  const notebook = new NotebookWorkspace(config.cells, config.tables ?? {});
  const store = new TerminalSessionStore<ConsoleState, Request>(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load({ requests: [], runs: [], status: 'idle' });
  const loop: EventLoop<ConsoleState, TerminalView, Request> = new EventLoop({
    initialState: checkpoint.state, initialRevision: checkpoint.revision, seenEventIds: checkpoint.seen_event_ids,
    reduce: async (state, event) => {
      const run = await answerRequest(notebook, event.value, openFolder(context.workspace).root());
      return { requests: [...state.requests, event.value], runs: [...state.runs, run], status: run.status };
    },
    view, step: fn => context.runtime.run(fn), onCommit: commit => store.commit(commit, loop.seenEventIds) });
  try {
    await runTerminalShell(loop, { input: context.io.input as never, output: context.io.output as never, color: context.io.color,
      event: (value, id) => ({ id, kind: 'request', value }), commands: {
        cells: { description: 'list notebook cells and dependencies', run: () => notebook.catalog()
          .map(cell => `${cell.id}  [${cell.engine}]  needs: ${cell.needs.join(', ') || 'none'}  ${cell.description}`).join('\n') },
        load: { description: 'FILE import or replace cells and tables', run: value => {
          if (!value) throw new Error('provide a notebook JSON path');
          const imported = notebook.importConfig(JSON.parse(readFileSync(resolve(context.workspace, value), 'utf8')));
          return `Loaded ${imported.cells.length} cells and ${imported.tables.length} tables.`;
        } },
        example: { description: 'show a useful first request', run: () => 'Which region has the largest total, and what is that total?' },
      } });
  } finally { await loop.close(); notebook.close(); }
  return 0;
}
