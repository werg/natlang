import { curriculumCase, evalCall, returnCall } from '../inline-curriculum/lib.mjs';
import { sha } from './common.mjs';
import { canonical } from '../advisory-file.mjs';

/** Shared directory-reducer scaffold used by annotation replay and provider labels. */
export function folderDecisionScaffold({ contract, entries, sourceSha, sourceSnapshot, sourceGroups,
  curriculumFamily, recordFamily, variant = 'strict-top-label-v2-ordinal-source-mapping', splitGroup,
  choiceGuidance }) {
  const rows = entries.map(entry => entry.source ?? entry.task ?? entry);
  const decisions = Object.fromEntries(entries.map((entry, index) => [rows[index].id,
    entry.decision ?? entry.task?.gold ?? entry.gold]));
  const ids = rows.map(row => row.id);
  const groups = [...new Set(sourceGroups ?? rows.map(row => row.group ?? row.group_id))];
  const first = rows[0];
  const batchKey = sha(canonical({ signature: contract.signature ?? contract,
    ids: ids.slice().sort(), groups: groups.slice().sort() })).slice(0, 24);
  const taskHasCriteria = contract.criteria !== undefined;
  const taskHasLevels = contract.levels !== undefined;
  if (rows.some(row => !Object.hasOwn(row, 'options') && contract.options === undefined))
    throw new Error('decision items need their own options or the task contract options');
  if (!taskHasCriteria && rows.some(row => Object.hasOwn(row, 'criteria')) &&
      !rows.every(row => Object.hasOwn(row, 'criteria')))
    throw new Error('criteria must be supplied for every item or in the shared task contract');
  if (!taskHasLevels && rows.some(row => Object.hasOwn(row, 'levels')) &&
      !rows.every(row => Object.hasOwn(row, 'levels')))
    throw new Error('levels must be supplied for every item or in the shared task contract');
  const hasCriteria = taskHasCriteria || rows.some(row => Object.hasOwn(row, 'criteria'));
  const hasLevels = taskHasLevels || rows.some(row => Object.hasOwn(row, 'levels'));
  const task = { family: first.family, kind: first.kind, options: contract.options, levels: contract.levels,
    labels: contract.labels, ...(contract.criteria !== undefined ? { criteria: contract.criteria } : {}), output_path: 'decisions.json' };
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
    : hasCriteria
      ? `Apply the supplied scoped criteria and return exactly one declared ${hasLevels ? 'option or level' : 'option'}. Do not return a probability distribution.`
      : `Choose exactly one answer from the supplied ${hasLevels ? 'options or levels' : 'options'}. Do not return a probability distribution.`;
  const criteriaFields = hasCriteria ? ', criteria' : '';
  const levelsFields = hasLevels ? ', levels' : '';
  const fieldInstructions = [
    'Read each item file for that item’s ID, state, question, and options. Preserve its options exactly; use task options only when that item has no options field.',
    hasCriteria
      ? 'Criteria are shared from task.json unless an item supplies its own criteria; use only criteria actually present in one of those files.'
      : 'Use the supplied options without a separate rubric; only criteria read from an item or task file are authoritative.',
    hasLevels
      ? 'Levels are shared from task.json unless an item supplies its own levels; an explicit empty task levels list means no levels are supplied.'
      : 'Choose from the supplied options; only levels read from an item or task file are part of the answer domain.',
    'Bind this item’s available fields before creating its inline natural-language function. Create that function inside the item’s callback so its question interpolation uses the bound question, and pass only supplied fields as separate arguments.',
    ...(choiceGuidance ? [choiceGuidance] : [])
  ].join(' ');
  const controller = `const task = await folder.file('task.json').readJson();
const decisions: ${outputType} = {};
for (const file of await folder.files('items/*.json')) {
 const item = await file.readJson();
 const caseId = item.id;
 const state = item.state;
 const question = item.question;
${hasCriteria ? " const criteria = Object.hasOwn(item, 'criteria') ? item.criteria : task.criteria;" : ''}
 const options = Object.hasOwn(item, 'options') ? item.options : task.options;
${hasLevels ? " const levels = Object.hasOwn(item, 'levels') ? item.levels : task.levels;" : ''}
 const decision = await nl<${returnType}>\`Question: \${question}\\nAnswer from the scoped state. ${kindInstruction}\`(caseId, state, question${criteriaFields}, options${levelsFields});
 decisions[caseId] = decision;
}
await folder.file(task.output_path).writeText(JSON.stringify(decisions));
return decisions;`;
  const childRefs = entries.map((entry, index) => {
    const itemHasCriteria = hasCriteria && (Object.hasOwn(rows[index], 'criteria') || Object.hasOwn(task, 'criteria'));
    const itemHasLevels = hasLevels && (Object.hasOwn(rows[index], 'levels') || Object.hasOwn(task, 'levels'));
    const fields = ['caseId', 'state', 'question', ...(itemHasCriteria ? ['criteria'] : []), 'options',
      ...(itemHasLevels ? ['levels'] : [])].join(', ');
    return { match: [ids[index]], calls: [
    evalCall(`console.log(JSON.stringify({ ${fields} }));`),
    returnCall(decisions[ids[index]]),
  ] }; });
  const record = curriculumCase({
    family: curriculumFamily ?? `teacher_decision_labels_${String(first.family).toLowerCase().replace(/[^a-z0-9]+/g, '_')}_${contract.kind}`,
    shape: `source_${sourceSha.slice(0, 12)}_${batchKey}`,
    variant, split: 'train', splitGroup: splitGroup ?? JSON.stringify(groups),
    slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    root: { name: 'reduce_decisions', kind: 'directory-reducer', args: {}, returns: outputType,
      instructions: `Read the task contract and every item in the directory. ${fieldInstructions} Compose the item question into the inline natural-language instruction at runtime. Pass case ID, state, question${criteriaFields}, options${levelsFields} separately. Do not combine fields into a serialized prompt variable or add captures for absent metadata. Save decisions.json and return that exact map.` },
    files: { 'types.ts': `export type Decision = ${union};\n` }, folderFiles,
    expectedFiles: { ...folderFiles, 'decisions.json': JSON.stringify(expected) }, expected,
    minimumSequence: ['read the runtime contract and directory items',
    `capture case ID, state, question${criteriaFields}, options${levelsFields} from the item and task files`,
      `compose the item question into the typed inline instruction and follow the ${contract.kind}-specific output rule`,
      'pass all captures separately for every item', 'save and return the exact aggregate map'],
    reference: { root: [evalCall(controller), returnCall(expected)], children: childRefs },
  });
  if (contract.ordinalTolerance) {
    const { levels, max_distance: maxDistance } = contract.ordinalTolerance;
    record.semantics.oracle = { level: 'normalized', normalization: 'ordinal-distance-record',
      ordinal_levels: levels, ordinal_distance_max: maxDistance };
    record.semantics.files_oracle = { compare: 'ordinal-distance-json-record', threshold: 1,
      ordinal_levels: levels, ordinal_distance_max: maxDistance };
  }
  record.family = recordFamily ?? `curriculum_teacher_decision_labels_${String(first.family).toLowerCase().replace(/[^a-z0-9]+/g, '_')}_${contract.kind}`;
  record.source = 'held-provider-decision-label-adapter';
  record.source_ids = ids;
  record.source_groups = groups;
  record.source_revisions = [sourceSnapshot, sourceSha];
  return record;
}
