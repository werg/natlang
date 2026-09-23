import type { Cell, CellResult, NotebookState, ConsoleEvent, File, ConsoleState, ViewBlock, TerminalView } from "./types.js";

export default function view(state: ConsoleState): TerminalView {
const run = state.runs.at(-1), request = state.requests.at(-1);
const blocks = run ? [
  { kind: 'text', text: `Request: ${request}` },
  { kind: 'status', text: `${run.status}: ${run.detail}`, tone: run.status === 'done' ? 'good' : 'warn' },
  { kind: 'text', text: run.answer || 'No answer was produced.' },
  { kind: 'table', columns: ['Cell', 'Status', 'Revision', 'Sample'], rows:
      run.results.map(result => [result.id, result.status, String(result.revision), result.sample || result.detail]) },
  ...(run.blocked.length ? [{ kind: 'list', items: run.blocked.map(id => `Blocked: ${id}`) }] : []),
] : [{ kind: 'text', text: 'A starter notebook is ready. Ask for a result, inspect /cells, or import your own notebook with /load FILE.', tone: 'muted' },
  { kind: 'list', items: ['Try: Which region has the largest total, and what is that total?', 'Use /help to discover setup and navigation commands.'] }];
return { title: 'Natlang Notebook Console', subtitle: `${state.runs.length} runs`, blocks,
  prompt: 'notebook> ', help: ['/help commands', '/cells catalog', '/load FILE import notebook', '/quit exit'] };
}
