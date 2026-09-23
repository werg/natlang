/** Evidence Console: citation-checked question answering over local sources in the terminal. */
import { join } from 'node:path';
import { EventLoop, TerminalSessionStore, runTerminalShell, type TargetContext, type TerminalView } from '@natlang/node';
import { EvidenceCollection, STARTER_EVIDENCE, answer, readEvidencePath, type EvidenceAnswer } from './index.js';

export type ConsoleState = { questions: string[], answers: EvidenceAnswer[], status: string };
export type ConsoleEvent = { id: string, kind: 'question', value: string };

export function view(state: ConsoleState): TerminalView {
  const latest = state.answers.at(-1), question = state.questions.at(-1);
  const blocks: TerminalView['blocks'] = latest ? [
    { kind: 'text', text: `Q: ${question}` },
    { kind: 'status', text: latest.status, tone: latest.status === 'citation-checked' ? 'good' : 'warn' },
    { kind: 'text', text: latest.answer },
    ...(latest.claims.length ? [{ kind: 'table' as const, columns: ['Claim', 'Source', 'Quote'],
      rows: latest.claims.map(claim => [claim.text, claim.span_id, claim.quote]) }] : []),
    ...(latest.gaps.length ? [{ kind: 'list' as const, items: latest.gaps }] : []),
  ] : [{ kind: 'text', text: 'Ready with a built-in guided collection. Ask a question, use /sources, or add your own material with /load PATH.', tone: 'muted' },
    { kind: 'list', items: ['Try: How does the console prevent fabricated citations?', 'Use /help to discover setup and navigation commands.'] }];
  return { title: 'Natlang Evidence Console', subtitle: `${state.answers.length} answered questions`, blocks,
    prompt: 'evidence> ', help: ['/help commands', '/sources collection', '/load PATH add evidence', '/quit exit'] };
}

export async function main(context: TargetContext): Promise<number> {
  const at = context.args.indexOf('--documents');
  const evidence = new EvidenceCollection(at >= 0 ? readEvidencePath(context.args[at + 1]!, context.workspace) : STARTER_EVIDENCE);
  const store = new TerminalSessionStore<ConsoleState, ConsoleEvent>(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load({ questions: [], answers: [], status: 'idle' });
  const loop: EventLoop<ConsoleState, TerminalView, ConsoleEvent> = new EventLoop({
    initialState: checkpoint.state, initialRevision: checkpoint.revision, seenEventIds: checkpoint.seen_event_ids,
    reduce: async (state, event) => {
      const result = await answer(evidence, event.value);
      return { questions: [...state.questions, event.value], answers: [...state.answers, result], status: result.status };
    },
    view, step: fn => context.runtime.run(fn), onCommit: commit => store.commit(commit, loop.seenEventIds) });
  try {
    await runTerminalShell(loop, { input: context.io.input as never, output: context.io.output as never, color: context.io.color,
      event: (value, id) => ({ id, kind: 'question', value }), commands: {
        sources: { description: 'list loaded evidence sources', run: () => evidence.catalog()
          .map(row => `${row.id}  ${row.paragraphs} paragraphs  ${row.characters} characters`).join('\n') },
        load: { description: 'PATH add a file, JSON collection, or directory', run: value => {
          if (!value) throw new Error('provide a path');
          const loaded = readEvidencePath(value, context.workspace);
          for (const document of loaded) evidence.update(document.id, document.text);
          return `Loaded ${loaded.length} source${loaded.length === 1 ? '' : 's'}: ${loaded.map(row => row.id).join(', ')}`;
        } },
        example: { description: 'show a useful first question', run: () => 'How does the console prevent fabricated citations?' },
      } });
  } finally { await loop.close(); }
  return 0;
}
