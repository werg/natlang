import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { worlds as authoredWorlds } from './semantic-iterate-worlds-v15-data.mjs';
import { makeIterateWorldCase, validateIterateWorlds } from './authored-iterate-source-builder.mjs';

const revision = 'authored-semantic-iterate-worlds-v15/12-soft-notes-readable';
const marker = text => `<|neuralese|>${text}<|/neuralese|>`;
const fieldsType = fields => `{ ${fields.map(field => `${field}: string`).join('; ')} }`;
const noteFor = (world, index) => {
  const pass = world.passes[index];
  const state = world.passStates[index];
  return `${pass.name}: ${pass.allowed_fields.map(field => `${field} ${JSON.stringify(state[field])}`).join('; ')}.`;
};

const rootTemplate = `const task = await folder.file('task.json').readJson();
type Draft = __FIELDS__;
type Progress = { pass: number; notes: Neuralese<string> };
const seed: Neuralese<(initialDraft: Draft) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({ taskInstruction: task.instruction,
  outputContract: JSON.stringify(task.output_contract) })\`Write brief readable prose notes saying the supplied draft is only a placeholder and no evidence has been reviewed. Do not infer any facts or a decision. Return prose only, not a JSON object.\`;
const initialNotes = await seed(task.initialDraft);
const revise = async (progress: Progress): Promise<Progress> => {
  const current = task.passes[progress.pass];
  const evidence = await folder.file(current.evidence_path);
  const step: Neuralese<(source: FileHandle, priorNotes: Neuralese<string>) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({
    taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract), decisionRule: task.output_contract.decision_rule,
    passName: current.name, passConstraint: current.constraint, allowedFields: JSON.stringify(current.allowed_fields) })\`Read the complete supplied current-pass source FileHandle and the priorNotes argument. PriorNotes is readable accumulated prose; inspect its actual text directly. Incorporate supported facts from this pass, preserve relevant earlier facts, and correct any superseded facts. Keep exact identifiers, amounts, counts, and qualifications. Return the complete updated prose notes only. Do not format notes as JSON or produce the final output object in this pass; the final interpreter will do that.\`;
  const notes = await step(evidence, progress.notes);
  return { pass: progress.pass + 1, notes };
};
const completed = await iterateOn(revise, { pass: 0, notes: initialNotes }).withLimit({ maxSteps: task.passes.length })
  .until(state => state.pass === task.passes.length);
const interpret: Neuralese<(notes: Neuralese<string>) => Promise<Draft>> = nl.with<Draft>({
  taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract),
  decisionRule: task.output_contract.decision_rule, outputPath: task.output_path })\`The notes argument is the complete accumulated readable prose. Read its actual text directly. Derive every declared field from those notes, applying the output contract, exact formats, and decision rule. Return exactly the final JSON object of the declared Draft type, with no extra text.\`;
const finalDraft = await interpret(completed.notes);
await folder.file(task.output_path).writeText(JSON.stringify(finalDraft));
const saved = await folder.file(task.output_path).readJson();
if (JSON.stringify(saved) !== JSON.stringify(finalDraft)) throw Error('saved draft differs');
return finalDraft;`;

function canonical(value) { return JSON.stringify(value); }
function applyV10Clarifications(world) {
  const copy = structuredClone(world);
  if (copy.slug === 'course_accommodation_version') {
    copy.evidence['pass-04-coordinator.md'] = copy.evidence['pass-04-coordinator.md'].replace(
      'signed approval CA-73 for AC-73A and BIO-42 version C8',
      'signed approval CA-73 for accommodation plan AC-73 and its signed addendum AC-73A, covering BIO-42 version C8');
  }
  if (copy.slug === 'vaccine_logger_correction') {
    copy.instruction += ' Only decision.json is an allowed write target; source evidence and other files are read-only.';
  }
  return copy;
}

export function makeSoftIterateCase(world, index) {
  const record = makeIterateWorldCase(world, index, { revision });
  const fields = Object.keys(world.fields);
  const task = JSON.parse(record.semantics.folder_files['task.json']);
  // Match V10's explicit vaccine file boundary. Other tasks rely on the common
  // decision.json output_path contract.
  if (world.slug === 'vaccine_logger_correction') task.output_contract.writable_paths = ['decision.json'];
  else delete task.output_contract.writable_paths;
  record.semantics.folder_files['task.json'] = canonical(task);
  record.semantics.expected_files['task.json'] = canonical(task);
  record.semantics.expected = structuredClone(world.passStates.at(-1));
  record.semantics.expected_files['decision.json'] = canonical(record.semantics.expected);
  const draftType = fieldsType(fields);
  record.id = record.id.replace(':evidence-scoped-iterative-derived-decision', ':evidence-scoped-soft-state-iterative-derived-decision-v1');
  const code = rootTemplate.replace('__FIELDS__', draftType);
  record.curriculum.shape = `v15-${world.slug}-four-pass-soft-notes-derive-and-revise`;
  record.curriculum.variant = 'evidence-scoped-iterative-soft-notes-to-crisp-draft/2-readable-notes';
  record.curriculum.minimum_sequence = [
    'read task instruction and output contract without loading future evidence',
    'create an initial readable prose note as a genuine soft block',
    'iterate four times with only the current FileHandle and previous readable Neuralese<string> notes',
    'carry each returned soft block into the next iteration through typed invocation arguments',
    'interpret the complete accumulated readable notes in one final NL child',
    `write and read back exactly ${fields.length} declared fields to decision.json`,
  ];
  record.curriculum.reference = {
    root: [['eval', { code }], ['return_result', { status: 'success', value: record.semantics.expected }]],
    children: [
      { match: 'Write brief readable prose notes saying the supplied draft is only a placeholder',
        calls: [['return_result', { status: 'success', value: marker('Starting state: the supplied draft is only a placeholder; no evidence has been reviewed.') }]],
        soft_output: { kind: 'Neuralese<string>', text: 'Starting state: the supplied draft is only a placeholder; no evidence has been reviewed.', next_argument: 'priorNotes' },
        expected_reads: [], expected_soft_input: 'priorNotes' },
      ...world.passes.map((pass, passIndex) => {
        let note = 'Starting state: the supplied draft is only a placeholder; no evidence has been reviewed.';
        for (let i = 0; i <= passIndex; i++) note += `\n${noteFor(world, i)}`;
        return {
          match: pass.evidence_path,
          calls: [['read_file', { path: pass.evidence_path }], ['return_result', { status: 'success', value: marker(note) }]],
          soft_output: { kind: 'Neuralese<string>', text: note, next_argument: passIndex === world.passes.length - 1 ? 'notes' : 'priorNotes' },
          expected_reads: [pass.evidence_path], ...(passIndex === world.passes.length - 1 ? {} : { expected_soft_input: 'priorNotes' }),
        };
      }),
      { match: 'The notes argument is the complete accumulated readable prose',
        calls: [['return_result', { status: 'success', value: record.semantics.expected }]],
        expected_reads: [], expected_soft_input: 'notes' },
    ],
  };
  record.semantics.files['reconcile_scoped_evidence.nl'] = `---\nargs: {}\nreturns: ${JSON.stringify(draftType)}\nkind: directory-reducer\n---\nRead task.json for the task contract. Iterate with Progress = { pass: number; notes: Neuralese<string> }. Seed readable prose notes; each pass child receives exactly its current evidence FileHandle and prior readable notes. Return updated prose notes only during the loop. After all four passes, one final NL child reads the complete notes and derives the exact declared output. Write only decision.json, read it back, and return the result.\n`;
  record.source_revisions = [revision];
  record.generation.generator = revision;
  record.generation.capture_contract = {
    task_contract: 'task instruction, output contract, decision rule, and current pass only',
    soft_state: 'Progress is { pass: number, notes: Neuralese<string> }; each pass child receives one current FileHandle and prior readable prose notes',
    final_interpreter: 'one final NL child receives and reads the complete accumulated readable prose notes, then derives the crisp Draft',
    evidence: 'each iterative child reads exactly the current pass FileHandle; no future-pass FileHandle or facts are supplied',
  };
  record.generation.source_quality = `${world.justified_revision.reason} Soft-state topology variant of V10; no new factual world.`;
  record.generation.soft_state_reference = 'Deterministic marker responses exercise real runtime port writes; not teacher observation or semantic truth.';
  return record;
}

export async function writeSoftIterateCandidate({ out }) {
  const worlds = authoredWorlds.map(applyV10Clarifications);
  validateIterateWorlds(worlds);
  const rows = worlds.map(makeSoftIterateCase);
  const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  const sourceSha = createHash('sha256').update(sourceText).digest('hex');
  const output = resolve(out);
  await mkdir(output, { recursive: false });
  await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
  const review = {
    schema: 'natlang.neuralese-source-quality-review/1', revision,
    source_cases_sha256: sourceSha,
    review_type: 'authored V15 soft-state prompt ergonomics correction; factual and topology review remains independent',
    admission_granted: false, model_calls: 0, provider_calls: 0, teacher_trajectories: 0,
    counts: { task_variants: 12, factual_source_groups: 12, train: 6, test: 6, passes: 48, justified_revisions: 12 },
    worlds: rows.map(row => ({ id: row.id, group: row.source_groups[0], split: row.split, expected: row.semantics.expected })),
  };
  await writeFile(resolve(output, 'source-quality-review.json'), JSON.stringify(review, null, 2) + '\n');
  const manifest = {
    schema: 'natlang.neuralese-semantic-iterate-v15-soft-readable/1', revision,
    source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
    scripted_proof: 'authored-scripted-proof-v79/runtime-reference-proof.json',
    parent_v10_source_sha256: '4a8f95f6ba6703437651c6c827b4042d89f536ab59cb2b7cc9496ed8199b7084',
    world_count: 12, task_variant_count: 12, train_count: 6, test_count: 6, pass_count: 48,
    source_groups_preserved: true, splits_preserved: true, final_gold_preserved: true,
    course_clarification: 'coordinator approval explicitly names the AC-73 plan and AC-73A addendum',
    vaccine_boundary: 'only decision.json is writable',
    topology: 'Progress {pass, notes: Neuralese<string>}; seed, four iterative note children, one final Draft interpreter; unchanged from V10',
    prompt_change: 'notes are explicitly readable accumulated prose; pass children return prose only; final interpreter reads actual notes directly; no opaque-content framing or verification subcalls',
    teacher_observations: 0, provider_calls: 0, training_admission: false, trace_admission: false,
  };
  await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  await writeFile(resolve(output, 'README.md'), `# V15 soft-state readable notes source\n\nFresh source-only prompt revision preserving V10 groups, splits, gold drafts, and soft-state graph topology. Per-pass notes are explicitly readable prose; the final interpreter reads the complete accumulated notes directly and returns the typed Draft.\n\nNo provider calls or teacher observations. Training and trace admission are false.\n\nSource SHA-256: ${sourceSha}.\n`);
  return { rows, sourceText, sourceSha, manifest };
}
