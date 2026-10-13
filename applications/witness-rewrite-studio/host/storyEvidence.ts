/** Exact external mechanics. No story policy or candidate selection lives here. */
import { createHash } from 'node:crypto';
import type { ExactCheck, ProposedEdit, SourceSnapshot, Witness } from '../types.js';
import type { StoryEvidence } from '../services.js';
const hash = (text: string): string => createHash('sha256').update(text).digest('hex');
function sourceChecks(snapshot: SourceSnapshot): ExactCheck[] {
  const ids = new Set<string>(), paths = new Set<string>();
  return snapshot.records.flatMap(record => {
    const valid = !ids.has(record.id) && !paths.has(record.path) && hash(record.text) === record.sha256;
    ids.add(record.id); paths.add(record.path);
    return [{ id: `source:${record.id}`, valid, detail: valid ? 'Unique identity and content hash match.' : 'Source identity or content hash mismatch.' }];
  });
}
function spans(snapshot: SourceSnapshot, edits: ProposedEdit[], allowed: string[]) {
  return edits.map(edit => {
    const record = snapshot.records.find(r => r.id === edit.sourceId && r.path === edit.sourcePath);
    const start = record?.text.indexOf(edit.oldQuote) ?? -1;
    const unique = start >= 0 && record?.text.indexOf(edit.oldQuote, start + 1) === -1;
    return {edit,record,start,end:start+edit.oldQuote.length,
      valid: !!record && allowed.includes(edit.sourcePath) && edit.oldQuote.length > 0 && unique};
  });
}
const storyEvidence: StoryEvidence = {
  checkWitnesses(snapshot, witnesses: Witness[]): ExactCheck[] {
    return [...sourceChecks(snapshot), ...witnesses.map((w, i) => {
      const record = snapshot.records.find(r => r.id === w.sourceId && r.path === w.path);
      const valid = !!record && w.quote.length > 0 && record.text.includes(w.quote);
      return { id: `witness:${i}:${w.sourceId}`, valid, detail: valid ? 'Exact source quotation matches.' : 'Source identity or exact quotation does not match.' };
    })];
  },
  checkEdits(snapshot, edits, allowedPaths) {
    const entries = spans(snapshot, edits, allowedPaths), checks = sourceChecks(snapshot);
    const seen = new Set<string>();
    for (const entry of entries) {
      const overlap = entries.some(other => other !== entry && other.edit.sourcePath === entry.edit.sourcePath &&
        entry.start < other.end && other.start < entry.end);
      const valid = entry.valid && !overlap && !seen.has(entry.edit.id);
      seen.add(entry.edit.id);
      checks.push({id:`edit:${entry.edit.id}`,valid,detail:valid ? 'Unique allowed original span; edits do not overlap.' : 'Identity, allowed path, unique span or nonoverlap check failed.'});
    }
    return checks;
  },
  applySnapshot(snapshot, edits, allowedPaths) {
    const failed = this.checkEdits(snapshot, edits, allowedPaths).find(check => !check.valid);
    if (failed) throw new Error(`Exact patch application failed: ${failed.id}: ${failed.detail}`);
    const entries = spans(snapshot, edits, allowedPaths);
    const records = snapshot.records.map(record => {
      const changes = entries.filter(e => e.edit.sourcePath === record.path).sort((a,b) => b.start-a.start);
      let text = record.text;
      for (const change of changes) text = text.slice(0,change.start)+change.edit.replacement+text.slice(change.end);
      return {...record,text,sha256:hash(text)};
    });
    return {revision:hash(JSON.stringify(records.map(({id,path,sha256})=>({id,path,sha256})))),records};
  },
};

/** Flat exports match the trusted collector service module binding. */
export const checkWitnesses: StoryEvidence['checkWitnesses'] = (snapshot, witnesses) => storyEvidence.checkWitnesses(snapshot, witnesses);
export const checkEdits: StoryEvidence['checkEdits'] = (snapshot, edits, allowedPaths) => storyEvidence.checkEdits(snapshot, edits, allowedPaths);
export const applySnapshot: StoryEvidence['applySnapshot'] = (snapshot, edits, allowedPaths) => storyEvidence.applySnapshot(snapshot, edits, allowedPaths);
