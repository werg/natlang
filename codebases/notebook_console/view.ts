/*---
engine: typescript-host
args:
  state: ConsoleState
returns: TerminalView
---*/
const state = args.state, run = state.runs.at(-1), request = state.requests.at(-1);
const blocks = run ? [
  { kind: 'text', text: `Request: ${request}` },
  { kind: 'status', text: `${run.status}: ${run.detail}`, tone: run.status === 'done' ? 'good' : 'warn' },
  { kind: 'text', text: run.answer || 'No answer was produced.' },
  { kind: 'table', columns: ['Cell', 'Status', 'Revision', 'Sample'], rows:
      run.results.map(result => [result.id, result.status, String(result.revision), result.sample || result.detail]) },
  ...(run.blocked.length ? [{ kind: 'list', items: run.blocked.map(id => `Blocked: ${id}`) }] : []),
] : [{ kind: 'text', text: 'Ask the notebook to compute or explain a result.', tone: 'muted' }];
return { title: 'Natlang Notebook Console', subtitle: `${state.runs.length} runs`, blocks,
  prompt: 'notebook> ', help: ['/refresh redraw', '/quit exit'] };
