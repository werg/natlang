import { curriculumCase, evalCall, returnCall } from '../inline-curriculum/lib.mjs';
import { sha } from './common.mjs';
import { canonical } from '../advisory-file.mjs';

/** Shared directory-reducer scaffold used by annotation replay and provider labels. */
export function folderDecisionScaffold({ contract, entries, sourceSha, sourceSnapshot, sourceGroups,
  curriculumFamily, recordFamily, variant = 'strict-top-label-v2-ordinal-source-mapping', splitGroup }) {
  const rows = entries.map(entry => entry.source ?? entry.task ?? entry);
  const decisions = Object.fromEntries(entries.map((entry, index) => [rows[index].id,
    entry.decision ?? entry.task?.gold ?? entry.gold]));
  const ids = rows.map(row => row.id);
  const groups = [...new Set(sourceGroups ?? rows.map(row => row.group ?? row.group_id))];
  const first = rows[0];
  const batchKey = sha(canonical({ signature: contract.signature ?? contract,
    ids: ids.slice().sort(), groups: groups.slice().sort() })).slice(0, 24);
  const task = { family: first.family, kind: first.kind, options: contract.options, levels: contract.levels,
    labels: contract.labels, criteria: contract.criteria, output_path: 'decisions.json' };
  const folderFiles = { 'task.json': JSON.stringify(task) };
  for (const row of rows) {
    const { answer: _answer, gold: _gold, gold_source: _goldSource, source_refs: _refs, identity_state_sha256: _identity,
      full_state_sha256: _full, ...visible } = row;
    folderFiles[`items/${row.id}.json`] = JSON.stringify(visible);
  }
  const expected = decisions;
  const union = contract.labels.map(JSON.stringify).join(' | ');
  const returnType = contract.kind === 'noul' ? 'boolean' : 'Decision';
  const outputType = contract.kind === 'noul' ? 'Record<string, boolean>' : 'Record<string, Decision>';
  const kindInstruction = contract.kind === 'noul'
    ? 'Return true when yes is at least as likely as no; otherwise return false. Do not return a numeric probability.'
    : 'Apply the scoped criteria and return exactly one declared option or level. Do not return a probability distribution.';
  const controller = `const task = await folder.file('task.json').readJson();
const decisions: ${outputType} = {};
for (const file of await folder.files('items/*.json')) {
 const item = await file.readJson();
 const caseId = item.id;
 const state = item.state;
 const question = item.question;
 const criteria = Object.hasOwn(item, 'criteria') ? item.criteria : task.criteria;
 const options = item.options ?? task.options ?? [];
 const levels = item.levels ?? task.levels ?? [];
 const decision = await nl<${returnType}>\`Question: \${question}\\nAnswer from the scoped state. Apply the scoped criteria. ${kindInstruction}\`(caseId, state, question, criteria, options, levels);
 decisions[caseId] = decision;
}
await folder.file(task.output_path).writeText(JSON.stringify(decisions));
return decisions;`;
  const childRefs = entries.map((entry, index) => ({ match: [ids[index]], calls: [
    evalCall('console.log(JSON.stringify({ caseId, state, question, criteria, options, levels }));'),
    returnCall(decisions[ids[index]]),
  ] }));
  const record = curriculumCase({
    family: curriculumFamily ?? `teacher_decision_labels_${String(first.family).toLowerCase().replace(/[^a-z0-9]+/g, '_')}_${contract.kind}`,
    shape: `source_${sourceSha.slice(0, 12)}_${batchKey}`,
    variant, split: 'train', splitGroup: splitGroup ?? JSON.stringify(groups),
    slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    root: { name: 'reduce_decisions', kind: 'directory-reducer', args: {}, returns: outputType,
      instructions: 'Read the task contract and every item in the directory. For each item, bind its case ID, state, question, criteria, options, and levels as separate scope values. Compose the item question into the inline natural-language instruction at runtime, and also pass question, state, criteria, options, and levels as separate arguments. Do not combine the fields into a serialized prompt variable. Save decisions.json and return that exact map.' },
    files: { 'types.ts': `export type Decision = ${union};\n` }, folderFiles,
    expectedFiles: { ...folderFiles, 'decisions.json': JSON.stringify(expected) }, expected,
    minimumSequence: ['read the runtime contract and directory items',
    'capture case ID, state, question, criteria, options and levels as separate scope values',
      `compose the item question into the typed inline instruction and follow the ${contract.kind}-specific output rule`,
      'pass all captures separately for every item', 'save and return the exact aggregate map'],
    reference: { root: [evalCall(controller), returnCall(expected)], children: childRefs },
  });
  record.family = recordFamily ?? `curriculum_teacher_decision_labels_${String(first.family).toLowerCase().replace(/[^a-z0-9]+/g, '_')}_${contract.kind}`;
  record.source = 'held-provider-decision-label-adapter';
  record.source_ids = ids;
  record.source_groups = groups;
  record.source_revisions = [sourceSnapshot, sourceSha];
  return record;
}
