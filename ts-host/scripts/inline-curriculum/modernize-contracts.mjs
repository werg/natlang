// Reviewed contract migrations preserve raw records and source identity; changed handoffs restart from the root.
import { commaqaQuestion } from './sources-ai2.mjs';
import { idempotentRetry } from './followup.mjs';
import { externalize } from './lib.mjs';

export function modernizeContracts(original) {
  const record = structuredClone(original), changes = [], family = record.curriculum?.family;
  const semantics = record.semantics;
  if (['commaqa_numeric', 'commaqa_question'].includes(family) && (record.curriculum.family_version ?? 1) < 3) {
    const sourceId = record.curriculum.evidence.background.join('\n').match(/(?:train|dev|test):\d+:\d+/)?.[0];
    if (!sourceId) throw new Error(`${record.id}: missing CommaQA source identity`);
    const [fresh] = commaqaQuestion(7, 0, family === 'commaqa_numeric' ? 'numeric' : 'explicit', sourceId);
    if (JSON.stringify(fresh.semantics.expected) !== JSON.stringify(semantics.expected))
      throw new Error(`${record.id}: changed CommaQA gold`);
    for (const [path, text] of Object.entries(fresh.semantics.files)) semantics.files[path] = text;
    semantics.services = { ...semantics.services, ...fresh.semantics.services };
    semantics.service_scopes = { ...semantics.service_scopes, ...fresh.semantics.service_scopes };
    record.curriculum.reference = fresh.curriculum.reference;
    record.curriculum.family_version = 3;
    record.generation = { ...record.generation, source_evidence_version: 3,
      ...(fresh.generation.source_annotation_repair ? { source_annotation_repair: fresh.generation.source_annotation_repair } : {}) };
    changes.push('correct source sports, literal retrieval and specialist schema');
  }
  if (family === 'idempotent_retry' && (record.curriculum.family_version ?? 1) < 2) {
    const [fresh] = idempotentRetry(7, 0);
    semantics.files[semantics.root] = fresh.semantics.files[fresh.semantics.root];
    record.curriculum.family_version = 2;
    changes.push('documented lost-ack confirmation');
  }
  if (family === 'event_retry' && Object.keys(semantics.files).some(path => path.endsWith('/board.ts'))) {
    externalize(semantics, family);
    if (!semantics.services?.board) throw new Error(`${record.id}: board was not externalized`);
    record.curriculum.family_version = 2;
    changes.push('board is an external service');
  }
  if (record.family === 'cb_highlighter') {
    for (const [path, source] of Object.entries(semantics.files)) {
      if (!path.endsWith('/split_source.ts')) continue;
      const fixed = source.replace(/(for \(const \w+ of )(\w+\.matchAll\(.+?\))\)(?= (?:if|names\.add|\{))/g, '$1[...$2])');
      if (fixed !== source) { semantics.files[path] = fixed; changes.push('finite matchAll iteration'); }
    }
  }
  if (changes.length) {
    record.id += ':contracts-v3';
    record.source_revisions = [...new Set([...record.source_revisions, 'natlang.contract_migration/3'])];
    record.generation = { ...record.generation, contract_migration: { version: 3, original_id: original.id,
      reset_handoff: !!record.handoff, changes } };
    // Replaying source edits or incorrect specialist questions under new contracts would seed incompatible state.
    delete record.handoff;
  }
  return { record, changes };
}
