import { nl, type Folder, type FolderHandle } from '@natlang/node';
import type { Ledger, Outcome } from './types.js';

type CaseNote = Is<string, 'one sentence'>;

/** Record settled requests in the desk's folder as one transaction: a case file for each, and the ledger's counts. */
export async function record(folder: Folder, outcomes: Outcome[]): Promise<Ledger | undefined> {
  const note = nl<(outcome: Outcome) => CaseNote>`Summarize outcome for the case file.`;
  const proposal = await folder.propose<[], Ledger>(async (draft: FolderHandle) => {
    const notes = await Promise.all(outcomes.map(outcome => note(outcome)));
    const ledger = await draft.file('ledger.json').readJson<Ledger>();
    for (const [index, outcome] of outcomes.entries()) {
      await draft.file(`cases/${outcome.id}.md`).writeText(`# ${outcome.id}: ${outcome.verdict}\n\n${notes[index]}\n`);
      ledger[outcome.verdict] += 1;
    }
    await draft.file('ledger.json').writeJson(ledger);
    return ledger;
  });
  // Accept only what was meant: one case file per outcome and the ledger, nothing else.
  const expected = new Set(['ledger.json', ...outcomes.map(outcome => `cases/${outcome.id}.md`)]);
  if (proposal.diff.changes.some(change => !expected.has(change.path))) return undefined;
  await folder.accept(proposal);
  return proposal.value;
}
