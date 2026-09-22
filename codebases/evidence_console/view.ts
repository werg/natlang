/*---
engine: typescript-host
args:
  state: ConsoleState
  files?: Dict<File>
returns: TerminalView
---*/
const state = args.state, answer = state.answers.at(-1), question = state.questions.at(-1);
const blocks = answer ? [
  { kind: 'text', text: `Q: ${question}` },
  { kind: 'status', text: answer.status, tone: answer.status === 'citation-checked' ? 'good' : 'warn' },
  { kind: 'text', text: answer.answer },
  ...(answer.claims.length ? [{ kind: 'table', columns: ['Claim', 'Source', 'Quote'],
    rows: answer.claims.map(claim => [claim.text, claim.span_id, claim.quote]) }] : []),
  ...(answer.gaps.length ? [{ kind: 'list', items: answer.gaps }] : []),
] : [{ kind: 'text', text: 'Ready with a built-in guided collection. Ask a question, use /sources, or add your own material with /load PATH.', tone: 'muted' },
  { kind: 'list', items: ['Try: How does the console prevent fabricated citations?', 'Use /help to discover setup and navigation commands.'] }];
return { title: 'Natlang Evidence Console', subtitle: `${state.answers.length} answered questions`, blocks,
  prompt: 'evidence> ', help: ['/help commands', '/sources collection', '/load PATH add evidence', '/quit exit'] };
