import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { worlds as authoredWorlds } from './semantic-iterate-worlds-v15-data.mjs';
import { makeSoftIterateCase } from './semantic-iterate-worlds-v15-soft-builder.mjs';
import { validateIterateWorlds } from './authored-iterate-source-builder.mjs';

export const GUIDED_SOFT_REVISION = 'authored-semantic-iterate-worlds-v15/14-guided-decision-now';
const canonical = value => JSON.stringify(value);
const marker = text => `<|neuralese|>${text}<|/neuralese|>`;

export const guidedRootTemplate = `const task = await folder.file('task.json').readJson();
type Draft = __FIELDS__;
type Progress = { pass: number; notes: Neuralese<string> };

const seed: Neuralese<(initialDraft: Draft) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({
  taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract)
})\`Create a brief readable prose note from the supplied initialDraft. State that it is only a placeholder and no evidence has been reviewed. Do not infer facts or a decision. Return Neuralese<string> prose only.\`;
const initialNotes = await seed(task.initialDraft);

const revise = async (progress: Progress): Promise<Progress> => {
  const current = task.passes[progress.pass];
  const evidence = await folder.file(current.evidence_path);
  const step: Neuralese<(source: FileHandle, priorNotes: Neuralese<string>) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({
    taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract),
    decisionRule: task.output_contract.decision_rule, passName: current.name,
    passConstraint: current.constraint, allowedFields: JSON.stringify(current.allowed_fields)
})\`Read only the supplied current-pass FileHandle and priorNotes. Do not open or infer from any other pass file. Preserve all supported earlier facts relevant to the decisionRule and outputContract; mark provisional findings when needed, and correct facts superseded by this pass. Keep historical events (what already happened) distinct from the requested decision (what must be decided now). Do not invent facts. Return the complete accumulated readable prose as Neuralese<string>; do not return a Draft or structured object.\`;
  const notes = await step(evidence, progress.notes);
  return { pass: progress.pass + 1, notes };
};

const completed = await iterateOn(revise, { pass: 0, notes: initialNotes })
  .withLimit({ maxSteps: task.passes.length })
  .until(state => state.pass === task.passes.length);

const interpret: Neuralese<(notes: Neuralese<string>) => Promise<Draft>> = nl.with<Draft>({
  taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract),
  decisionRule: task.output_contract.decision_rule, outputPath: task.output_path
})\`Read the complete accumulated Neuralese<string> notes. Determine the requested decision now by applying exactly the explicit decisionRule to the supported facts and outputContract. Historical events are evidence for that decision, not a substitute for making it. Do not require evidence that the requested decision has already been executed, and do not add eligibility, authorization, or other prerequisites absent from the explicit decisionRule. Do not invent facts; follow the rule's stated handling of uncertainty. Derive every declared final Draft field from the notes and contract. Return exactly the declared Draft, with no extra text.\`;
const finalDraft = await interpret(completed.notes);
await folder.file(task.output_path).writeText(JSON.stringify(finalDraft));
const saved = await folder.file(task.output_path).readJson();
if (JSON.stringify(saved) !== JSON.stringify(finalDraft)) throw Error('saved draft differs');
return finalDraft;`;

function clarifiedWorld(world) {
  const copy = structuredClone(world);
  if (copy.slug === 'course_accommodation_version') {
    copy.evidence['pass-04-coordinator.md'] = copy.evidence['pass-04-coordinator.md'].replace(
      'signed approval CA-73 for AC-73A and BIO-42 version C8',
      'signed approval CA-73 for accommodation plan AC-73 and its signed addendum AC-73A, covering BIO-42 version C8');
  }
  if (copy.slug === 'vaccine_logger_correction')
    copy.instruction += ' Only decision.json is an allowed write target; source evidence and other files are read-only.';
  return copy;
}

function finalDraftType(world) {
  const fields = Object.keys(world.fields);
  return `{ ${fields.map(field => {
    const metadata = world.field_enums?.[field];
    const values = Array.isArray(metadata) ? metadata : metadata?.final;
    const type = Array.isArray(values) && values.length ? values.map(value => JSON.stringify(value)).join(' | ') : 'string';
    return `${field}: ${type}`;
  }).join('; ')} }`;
}

export function makeGuidedSoftIterateCase(world, index) {
  const preparedWorld = clarifiedWorld(world);
  const record = makeSoftIterateCase(preparedWorld, index);
  const fields = Object.keys(world.fields);
  const task = JSON.parse(record.semantics.folder_files['task.json']);
  task.output_contract.carry_forward = 'During note passes, preserve supported earlier facts relevant to the decisionRule and outputContract; mark uncertainty and correct facts superseded by current evidence. Keep historical events distinct from the requested decision to be made now. Do not invent facts. The loop carries Neuralese<string> notes, not a Draft. The final interpreter applies exactly the explicit decisionRule to supported accumulated facts and the outputContract; it does not require proof that the requested decision was already executed or add unstated prerequisites.';
  if (!task.output_contract.final_field_enums) delete task.output_contract.enum_contract;
  else task.output_contract.enum_contract = 'final_field_enums constrain only the final Draft; they do not constrain the accumulated Neuralese<string> notes.';
  record.semantics.folder_files['task.json'] = canonical(task);
  record.semantics.expected_files['task.json'] = canonical(task);
  record.semantics.expected = structuredClone(preparedWorld.passStates.at(-1));
  record.semantics.expected_files['decision.json'] = canonical(record.semantics.expected);

  const code = guidedRootTemplate.replace('__FIELDS__', finalDraftType(preparedWorld));
  record.id = record.id.replace(':evidence-scoped-soft-state-iterative-derived-decision-v1', ':evidence-scoped-guided-soft-state-derived-decision-v1');
  record.source_revisions = [GUIDED_SOFT_REVISION];
  record.generation.generator = GUIDED_SOFT_REVISION;
  record.generation.capture_contract = {
    task_contract: 'task instruction, output contract, explicit decision rule, and current pass only',
    soft_state: 'Progress is { pass: number, notes: Neuralese<string> }; the seed and each pass return Neuralese<string>',
    seed: 'one typed Neuralese<string> seed child receives initialDraft and creates a placeholder note without facts',
    iterative_children: 'four typed nl.with<Neuralese<string>> children; each receives only its current FileHandle and the prior Neuralese<string> notes',
    final_interpreter: 'one typed Draft interpreter decides the requested action now from supported accumulated facts by applying exactly the explicit decisionRule and outputContract; no already-executed-action prerequisite or added conditions',
    evidence: 'root scaffold opens no evidence before the loop; each step opens only its current pass FileHandle',
  };
  record.generation.source_quality = `${preparedWorld.justified_revision.reason} Guided soft-state topology revision; no new factual world.`;
  record.generation.soft_state_reference = 'Authored reference exercises typed Neuralese writer-to-argument-to-reader edges; scripted evidence is not teacher observation or semantic truth.';
  record.curriculum.shape = `v15-${world.slug}-four-pass-guided-soft-notes`;
  record.curriculum.variant = 'evidence-scoped-guided-neuralese-soft-notes-to-crisp-draft/1';
  record.curriculum.minimum_sequence = [
    'read task contract and initial placeholder without opening any evidence file',
    'create one typed Neuralese<string> placeholder note through a seed child',
    'iterate four times with persisted pass state and only the current evidence FileHandle',
    'carry each Neuralese<string> output through the typed priorNotes argument',
    'derive the final Draft in one typed interpreter from all accumulated Neuralese<string> notes',
    `write and read back exactly ${fields.length} declared fields to decision.json`,
  ];
  record.curriculum.reference.root = [['eval', { code }], ['return_result', { status: 'success', value: record.semantics.expected }]];
  const initialNote = 'Starting state: the supplied draft is only a placeholder; no evidence has been reviewed.';
  record.curriculum.reference.children = [
    {
      match: 'Create a brief readable prose note from the supplied initialDraft',
      calls: [['return_result', { status: 'success', value: marker(initialNote) }]],
      soft_output: { kind: 'Neuralese<string>', text: initialNote, next_argument: 'priorNotes' },
      expected_reads: [],
    },
    ...preparedWorld.passes.map((pass, passIndex) => {
      let note = initialNote;
      for (let i = 0; i <= passIndex; i++)
        note += `\n${preparedWorld.passes[i].name}: ${preparedWorld.passes[i].allowed_fields.map(field => `${field} ${JSON.stringify(preparedWorld.passStates[i][field])}`).join('; ')}.`;
      return {
        match: pass.evidence_path,
        calls: [['read_file', { path: pass.evidence_path }], ['return_result', { status: 'success', value: marker(note) }]],
        soft_output: { kind: 'Neuralese<string>', text: note, next_argument: passIndex === preparedWorld.passes.length - 1 ? 'notes' : 'priorNotes' },
        expected_reads: [pass.evidence_path],
        expected_soft_input: 'priorNotes',
      };
    }),
    {
      match: 'Read the complete accumulated Neuralese<string> notes',
      calls: [['return_result', { status: 'success', value: record.semantics.expected }]],
      expected_reads: [], expected_soft_input: 'notes',
    },
  ];
  record.curriculum.reference.children.at(-2).expected_soft_input = 'priorNotes';
  record.curriculum.reference.children.at(-1).expected_soft_input = 'notes';
  record.semantics.files['reconcile_scoped_evidence.nl'] = `---\nargs: {}\nreturns: ${JSON.stringify(finalDraftType(preparedWorld))}\nkind: directory-reducer\n---\nUse this scaffold for the task. Keep the runtime's existing Neuralese<T> type; do not redefine or alias it. Read task.json only before starting. Do not read evidence until the current iterateOn step. Each step opens exactly task.passes[progress.pass].evidence_path, passes that FileHandle and prior Neuralese<string> notes to nl.with<Neuralese<string>>, and returns the updated Neuralese<string> notes. The final nl.with<Draft> reads all accumulated notes and returns the exact final Draft.\n\n\`\`\`ts\n${code}\n\`\`\`\n`;
  return record;
}

export async function writeGuidedSoftIterateCandidate({ out }) {
  const worlds = authoredWorlds;
  validateIterateWorlds(worlds);
  const rows = worlds.map((world, index) => makeGuidedSoftIterateCase(world, index));
  const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  const sourceSha = createHash('sha256').update(sourceText).digest('hex');
  const output = resolve(out);
  await mkdir(output, { recursive: false });
  await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
  const review = {
    schema: 'natlang.neuralese-source-quality-review/1', revision: GUIDED_SOFT_REVISION,
    source_cases_sha256: sourceSha,
    review_type: 'scaffold-assisted V15 soft-state provenance/topology source revision; not unassisted teacher success',
    admission_granted: false, model_calls: 0, provider_calls: 0, teacher_trajectories: 0,
    counts: { task_variants: 12, factual_source_groups: 12, train: 6, test: 6, passes: 48, justified_revisions: 12 },
    worlds: rows.map(row => ({ id: row.id, group: row.source_groups[0], split: row.split, expected: row.semantics.expected })),
  };
  await writeFile(resolve(output, 'source-quality-review.json'), JSON.stringify(review, null, 2) + '\n');
  const manifest = {
    schema: 'natlang.neuralese-semantic-iterate-v15-guided-soft/1', revision: GUIDED_SOFT_REVISION,
    source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
    scripted_proof: 'authored-scripted-proof-v83/runtime-reference-proof.json',
    frozen_v83_runtime_manifest_sha256: 'dda06a0818b2508f88bb0c28012edf2bf9c53197771a84aedff90bb3bacaea02',
    parent_v14_guided_source_sha256: 'd32da4e94b88632d98f9ee1ee0706310a1f5ed3760ecb2c9ab2fb23877bc7831',
    world_count: 12, task_variant_count: 12, train_count: 6, test_count: 6, pass_count: 48,
    source_groups_preserved: true, splits_preserved: true, final_gold_preserved: true,
    course_clarification: 'coordinator approval explicitly names the AC-73 plan and AC-73A addendum',
    vaccine_boundary: 'only decision.json is writable',
    topology: 'one Neuralese<string> seed writer, four Neuralese<string> pass writers carried as priorNotes refs, and one typed final Draft interpreter',
    prompt_change: 'visible authored eval scaffold with true Neuralese types, no evidence pre-read, current-pass-only FileHandle, notes preserve decision-relevant facts and separate history from the requested decision, final interpreter applies exactly the explicit decisionRule now without added prerequisites',
    proof_scope: 'frozen V83 compiled runtime with deterministic authored referenceDriver; scaffold-assisted source graph proof, not unassisted teacher success',
    unassisted_teacher_success: false,
    teacher_observations: 0, provider_calls: 0, training_admission: false, trace_admission: false,
  };
  await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  await writeFile(resolve(output, 'README.md'), `# V15 guided Neuralese soft-state source\n\nFresh scaffold-assisted source revision. It preserves the V13 readable source groups, splits, gold outputs, course AC-73 plus AC-73A clarification, and vaccine decision.json write boundary. The authored root instruction visibly supplies the typed eval scaffold without embedding per-world answers or notes.\n\nCurrent-pass notes preserve supported decision-relevant facts, distinguish historical events from the requested decision, and record uncertainty without inventing facts. The final interpreter applies exactly the explicit decisionRule to accumulated facts to determine the requested decision now; it does not require evidence that the decision was already executed or add unstated prerequisites.\n\nThe scaffold calls a Neuralese<string> seed, four Neuralese<string> pass children, and one typed final Draft interpreter. Each pass opens only its current FileHandle.\n\nThe frozen V83 CPU proof is recorded separately. This is scaffold-assisted source graph validation, not unassisted teacher success, semantic truth, teacher observation, or training admission. No provider or teacher calls were made while building this artifact.\n\nSource SHA-256: ${sourceSha}.\n`);
  return { rows, sourceText, sourceSha, manifest };
}
