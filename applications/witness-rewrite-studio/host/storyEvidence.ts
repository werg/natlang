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

/** Runtime shape checks teach the declared positional call without exposing a raw JS TypeError. */
const recordShape = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const stringList = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string');
function snapshotShape(value: unknown): value is SourceSnapshot {
  return recordShape(value) && typeof value.revision === 'string' && Array.isArray(value.records) && value.records.every(record =>
    recordShape(record) && ['id', 'path', 'text', 'sha256'].every(key => typeof record[key] === 'string'));
}
function witnessShape(value: unknown): value is Witness {
  return recordShape(value) && ['sourceId', 'path', 'quote'].every(key => typeof value[key] === 'string');
}
function editShape(value: unknown): value is ProposedEdit {
  return recordShape(value) && ['id', 'sourceId', 'sourcePath', 'oldQuote', 'replacement', 'sceneId'].every(key => typeof value[key] === 'string') &&
    stringList(value.removedFactIds) && stringList(value.addedMeanings) && Array.isArray(value.witnesses) && value.witnesses.every(witnessShape);
}
export function checkWitnesses(snapshot: SourceSnapshot, witnesses: Witness[]): ExactCheck[] {
  if (!snapshotShape(snapshot) || !Array.isArray(witnesses) || !witnesses.every(witnessShape))
    throw new Error('Call storyEvidence.checkWitnesses(snapshot, witnesses) with a SourceSnapshot and a Witness[] array.');
  return storyEvidence.checkWitnesses(snapshot, witnesses);
}
export function checkEdits(snapshot: SourceSnapshot, edits: ProposedEdit[], allowedPaths: string[]): ExactCheck[] {
  if (!snapshotShape(snapshot) || !Array.isArray(edits) || !edits.every(editShape) || !stringList(allowedPaths))
    throw new Error('Call storyEvidence.checkEdits(snapshot, edits, allowedPaths) with a SourceSnapshot, a ProposedEdit[] array and a string[] path array.');
  return storyEvidence.checkEdits(snapshot, edits, allowedPaths);
}
export function applySnapshot(snapshot: SourceSnapshot, edits: ProposedEdit[], allowedPaths: string[]): SourceSnapshot {
  if (!snapshotShape(snapshot) || !Array.isArray(edits) || !edits.every(editShape) || !stringList(allowedPaths))
    throw new Error('Call storyEvidence.applySnapshot(snapshot, edits, allowedPaths) with a SourceSnapshot, a ProposedEdit[] array and a string[] path array.');
  return storyEvidence.applySnapshot(snapshot, edits, allowedPaths);
}
