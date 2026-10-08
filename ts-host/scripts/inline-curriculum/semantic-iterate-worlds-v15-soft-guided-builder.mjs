import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { worlds as authoredWorlds } from './semantic-iterate-worlds-v15-data.mjs';
import { makeSoftIterateCase } from './semantic-iterate-worlds-v15-soft-builder.mjs';
import { validateIterateWorlds } from './authored-iterate-source-builder.mjs';

export const GUIDED_SOFT_REVISION = 'authored-semantic-iterate-worlds-v15/14-guided-decision-now';
const canonical = value => JSON.stringify(value);
const marker = text => `<|neuralese|>${text}<|/neuralese|>`;

export const guidedRootTemplate = `const task = await folder.file('task.json').readJson();
type Draft = __FIELDS__;
__INITIAL_DRAFT_DECL__
type Progress = { pass: number; notes: Neuralese<string> };

const seed: Neuralese<(initialDraft: __INITIAL_DRAFT_TYPE__) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({
  taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract)
})\`Create a brief readable prose note from the supplied initialDraft. State that it is only a placeholder and no evidence has been reviewed. Do not infer facts or a decision. Return Neuralese<string> prose only.\`;
const initialNotes = await seed(task.initialDraft);

const revise = async (progress: Progress): Promise<Progress> => {
  __STEP_SETUP__
  __STEP_DECL__
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
    copy.evidence['pass-04-coordinator.md'] = copy.evidence['pass-04-coordinator.md'].replace(
      'signed approval CA-73 for AC-73 plan and AC-73A addendum covering BIO-42 version C8',
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

function initialDraftType(world) {
  const fields = Object.keys(world.fields);
  return `{ ${fields.map(field => {
    const metadata = world.field_enums?.[field];
    const values = Array.isArray(metadata) ? metadata : metadata?.intermediate;
    const type = Array.isArray(values) && values.length ? values.map(value => JSON.stringify(value)).join(' | ') : 'string';
    return `${field}: ${type}`;
  }).join('; ')} }`;
}

function summarizeObservedEvidence(path, text) {
  const scores = [...text.matchAll(/([A-Z]{3}-\d+[A-D]) has priority score (\d+)/g)]
    .map(([, id, score]) => `${id}=${score}`);
  if (scores.length) return `Derived from this observed file, priority facts are ${scores.join('; ')}.`;

  const auditRows = [...text.matchAll(/([A-Z]{3}-\d+[A-D])\s+([^.]+)\./g)]
    .map(([, id, fact]) => `${id}: ${fact.trim()}`);
  if (/eligibility|audit/i.test(path) && auditRows.length)
    return `Derived from this observed file, candidate audit statements are ${auditRows.join('; ')}.`;

  const signed = /Authorized reviewer signed ([A-Z]{3}-\d+) for disposition on ([^.]+)\./.exec(text);
  if (signed) return `Derived from this observed file, reviewer signature is recorded for ${signed[1]} on ${signed[2]}.`;
  const unsigned = /No authorized reviewer has signed ([A-Z]{3}-\d+) by ([^.]+)\./.exec(text);
  if (unsigned) return `Derived from this observed file, no authorized signature is recorded for ${unsigned[1]} by ${unsigned[2]}.`;

  const review = /opened review ([A-Z]{3}-\d+) for ([^.]+)\. The decision deadline is ([^.]+)\./.exec(text);
  if (review) return `Derived from this observed file, review ${review[1]} concerns ${review[2]} with deadline ${review[3]}.`;
  return 'No extracted summary; retain and use the complete source text above.';
}

export function makeGuidedSoftIterateCase(world, index, { revision = GUIDED_SOFT_REVISION, shapeVersion = 'v15', savedWith = false } = {}) {
  const preparedWorld = clarifiedWorld(world);
  const record = makeSoftIterateCase(preparedWorld, index);
  const fields = Object.keys(world.fields);
  const task = JSON.parse(record.semantics.folder_files['task.json']);
  task.output_contract.evidence_files = 'The four pass-*.md files are read-only evidence inputs. Preserve their complete text exactly; do not rewrite, paraphrase, or replace them. Only decision.json is a writable output.';
  task.instruction += '\n\nThe four pass-*.md records are read-only evidence inputs. Preserve their complete contents exactly. Do not rewrite or paraphrase them; write only decision.json.';
  task.output_contract.carry_forward = 'During note passes, preserve supported earlier facts relevant to the decisionRule and outputContract; mark uncertainty and correct facts superseded by current evidence. Keep historical events distinct from the requested decision to be made now. Do not invent facts. The loop carries Neuralese<string> notes, not a Draft. The final interpreter applies exactly the explicit decisionRule to supported accumulated facts and the outputContract; it does not require proof that the requested decision was already executed or add unstated prerequisites.';
  const finalEnums = Object.fromEntries(Object.entries(preparedWorld.field_enums ?? {}).map(([field, metadata]) => [field, Array.isArray(metadata) ? metadata : metadata.final]));
  if (Object.keys(finalEnums).length) {
    task.output_contract.intermediate_field_enums = Object.fromEntries(Object.entries(preparedWorld.field_enums).map(([field, metadata]) => [field, Array.isArray(metadata) ? metadata : metadata.intermediate]));
    task.output_contract.final_field_enums = finalEnums;
    task.output_contract.enum_contract = 'intermediate_field_enums constrain only the initialDraft placeholder; final_field_enums constrain only the final Draft. Neither constrains the accumulated Neuralese<string> notes. Enum fields in the final Draft must contain exactly one listed literal, with no explanation or added units.';
  } else {
    delete task.output_contract.final_field_enums;
    delete task.output_contract.enum_contract;
  }
  record.semantics.folder_files['task.json'] = canonical(task);
  record.semantics.expected_files['task.json'] = canonical(task);
  record.semantics.expected = structuredClone(preparedWorld.passStates.at(-1));
  record.semantics.expected_files['decision.json'] = canonical(record.semantics.expected);

  const enumGuidance = 'Follow every declared field format exactly. For enum fields, return one bare listed literal only; do not add a prose explanation, unit label, or other text unless that field\'s declared format explicitly requires it. ';
  const stepType = '(source: FileHandle, priorNotes: Neuralese<string>) => Promise<Neuralese<string>>';
  const stepPrompt = savedWith
    ? 'Use the captured current pass context, including current.taskInstruction, current.outputContract, current.decisionRule, current.passName, current.passConstraint, and current.allowedFields. Read only the supplied current-pass FileHandle and priorNotes. Do not open or infer from any other pass file. Preserve supported earlier facts relevant to the decisionRule and outputContract; mark provisional findings when needed, and correct facts superseded by this pass. Keep historical events distinct from the requested decision. Do not invent facts. Return complete accumulated readable prose as Neuralese<string>, not a Draft or structured object.'
    : 'Read only the supplied current-pass FileHandle and priorNotes. Do not open or infer from any other pass file. Preserve all supported earlier facts relevant to the decisionRule and outputContract; mark provisional findings when needed, and correct facts superseded by this pass. Keep historical events (what already happened) distinct from the requested decision (what must be decided now). Do not invent facts. Return the complete accumulated readable prose as Neuralese<string>; do not return a Draft or structured object.';
  const stepCaptures = `{
    taskInstruction: task.instruction, outputContract: JSON.stringify(task.output_contract),
    decisionRule: task.output_contract.decision_rule, passName: current.name,
    passConstraint: current.constraint, allowedFields: JSON.stringify(current.allowed_fields)
  }`;
  const stepSetup = savedWith
    ? `const current = { pass: task.passes[progress.pass], taskInstruction: task.instruction,
    outputContract: JSON.stringify(task.output_contract), decisionRule: task.output_contract.decision_rule,
    passName: task.passes[progress.pass].name, passConstraint: task.passes[progress.pass].constraint,
    allowedFields: JSON.stringify(task.passes[progress.pass].allowed_fields) };
  const evidence = await folder.file(current.pass.evidence_path);`
    : `const current = task.passes[progress.pass];
  const evidence = await folder.file(current.evidence_path);`;
  const stepDeclaration = savedWith
    ? `const stepTemplate = nl<${stepType}>\`${stepPrompt}\`;
  const step = stepTemplate.with({ current });`
    : `const step: Neuralese<${stepType}> = nl.with<Neuralese<string>>(${stepCaptures})\`${stepPrompt}\`;`;
  const code = guidedRootTemplate
    .replace('__STEP_SETUP__', stepSetup)
    .replace('__STEP_DECL__', stepDeclaration)
    .replace('__FIELDS__', finalDraftType(preparedWorld))
    .replace('__INITIAL_DRAFT_DECL__\n', `type InitialDraft = ${initialDraftType(preparedWorld)};\n`)
    .replace('__INITIAL_DRAFT_TYPE__', 'InitialDraft')
    .replace('Derive every declared final Draft field', `${enumGuidance}Derive every declared final Draft field`);
  const hasFinalEnums = Object.keys(finalEnums).length > 0;
  record.id = record.id.replace(':evidence-scoped-soft-state-iterative-derived-decision-v1', `:evidence-scoped-guided-soft-state-derived-decision-${hasFinalEnums ? 'v2' : 'v1'}`);
  record.source_revisions = [revision];
  record.generation.generator = revision;
  record.generation.capture_contract = {
    task_contract: 'task instruction, output contract, explicit decision rule, and current pass only',
    soft_state: 'Progress is { pass: number, notes: Neuralese<string> }; the seed and each pass return Neuralese<string>',
    seed: 'one typed Neuralese<string> seed child receives initialDraft and creates a placeholder note without facts',
    iterative_children: 'four typed nl.with<Neuralese<string>> children; each receives only its current FileHandle and the prior Neuralese<string> notes',
    final_interpreter: shapeVersion === 'v16' ? 'one typed literal-union Draft interpreter decides now by applying exactly the explicit decisionRule and outputContract; it emits exact field formats and bare listed enum values, without added conditions' : 'one typed Draft interpreter decides the requested action now from supported accumulated facts by applying exactly the explicit decisionRule and outputContract; no already-executed-action prerequisite or added conditions',
    evidence: 'root scaffold opens no evidence before the loop; each step opens only its current pass FileHandle',
  };
  record.generation.source_quality = `${preparedWorld.justified_revision.reason} Guided soft-state topology revision; no new factual world.`;
  record.generation.soft_state_reference = 'Authored reference exercises typed Neuralese writer-to-argument-to-reader edges; scripted evidence is not teacher observation or semantic truth.';
  record.curriculum.shape = `${shapeVersion}-${world.slug}-four-pass-guided-soft-notes`;
  record.curriculum.variant = hasFinalEnums ? 'evidence-scoped-guided-neuralese-soft-notes-to-enum-constrained-draft/2' : 'evidence-scoped-guided-neuralese-soft-notes-to-crisp-draft/1';
  record.curriculum.minimum_sequence = [
    'read task contract and initial placeholder without opening any evidence file',
    'create one typed Neuralese<string> placeholder note through a seed child',
    'iterate four times with persisted pass state and only the current evidence FileHandle',
    'carry each Neuralese<string> output through the typed priorNotes argument',
    'derive the final Draft from accumulated notes, matching each field format and final enum literal exactly',
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
      for (let i = 0; i <= passIndex; i++) {
        const observedPass = preparedWorld.passes[i];
        const observedText = preparedWorld.evidence[observedPass.evidence_path];
        const summary = summarizeObservedEvidence(observedPass.evidence_path, observedText);
        note += `\nObserved source ${observedPass.evidence_path} (complete text): ${observedText}\nSummary derived only from that file: ${summary}`;
      }
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

export async function writeGuidedSoftIterateCandidate({ out, worlds = authoredWorlds, revision = GUIDED_SOFT_REVISION, shapeVersion = 'v15' }) {
  validateIterateWorlds(worlds);
  const enumHelperPath = resolve(import.meta.dirname, 'semantic-iterate-worlds-v15-enum-data.mjs');
  const enumHelperSha = shapeVersion === 'v16' ? createHash('sha256').update(await readFile(enumHelperPath)).digest('hex') : undefined;
  const rows = worlds.map((world, index) => makeGuidedSoftIterateCase(world, index, { revision, shapeVersion }));
  const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  const sourceSha = createHash('sha256').update(sourceText).digest('hex');
  const output = resolve(out);
  await mkdir(output, { recursive: false });
  await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
  const review = {
    schema: 'natlang.neuralese-source-quality-review/1', revision,
    source_cases_sha256: sourceSha,
    review_type: shapeVersion === 'v16' ? 'scaffold-assisted V16 literal-union final Draft ergonomics revision; not unassisted teacher success' : 'scaffold-assisted V15 soft-state provenance/topology source revision; not unassisted teacher success',
    admission_granted: false, model_calls: 0, provider_calls: 0, teacher_trajectories: 0,
    counts: { task_variants: 12, factual_source_groups: 12, train: 6, test: 6, passes: 48, justified_revisions: 12 },
    worlds: rows.map(row => ({ id: row.id, group: row.source_groups[0], split: row.split, expected: row.semantics.expected })),
  };
  await writeFile(resolve(output, 'source-quality-review.json'), JSON.stringify(review, null, 2) + '\n');
  const manifest = {
    schema: `natlang.neuralese-semantic-iterate-${shapeVersion}-guided-soft/1`, revision,
    source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
    scripted_proof: 'authored-scripted-proof-v83/runtime-reference-proof.json',
    frozen_v83_runtime_manifest_sha256: 'dda06a0818b2508f88bb0c28012edf2bf9c53197771a84aedff90bb3bacaea02',
    parent_v14_guided_source_sha256: 'd32da4e94b88632d98f9ee1ee0706310a1f5ed3760ecb2c9ab2fb23877bc7831',
    ...(shapeVersion === 'v16' ? { enum_contract_source: 'ts-host/scripts/inline-curriculum/semantic-iterate-worlds-v15-enum-data.mjs', enum_contract_source_sha256: enumHelperSha } : {}),
    world_count: 12, task_variant_count: 12, train_count: 6, test_count: 6, pass_count: 48,
    source_groups_preserved: true, splits_preserved: true, final_gold_preserved: true,
    ...(shapeVersion === 'v16' ? { enum_contracts: 'Known finite final field sets are explicit literal unions in the typed Draft and task output contract; intermediate placeholders use a separate InitialDraft union; prose notes remain Neuralese<string>.' } : {}),
    course_clarification: 'coordinator approval explicitly names the AC-73 plan and AC-73A addendum',
    vaccine_boundary: 'only decision.json is writable',
    topology: 'one Neuralese<string> seed writer, four Neuralese<string> pass writers carried as priorNotes refs, and one typed final Draft interpreter',
    prompt_change: shapeVersion === 'v16' ? 'visible authored eval scaffold with true Neuralese types, no evidence pre-read, current-pass-only FileHandle, notes preserve decision-relevant facts and separate history from the requested decision, final interpreter applies exactly the explicit decisionRule now without added prerequisites and uses literal-union Draft types plus exact field-format instructions' : 'visible authored eval scaffold with true Neuralese types, no evidence pre-read, current-pass-only FileHandle, notes preserve decision-relevant facts and separate history from the requested decision, final interpreter applies exactly the explicit decisionRule now without added prerequisites',
    proof_scope: 'frozen V83 compiled runtime with deterministic authored referenceDriver; scaffold-assisted source graph proof, not unassisted teacher success',
    unassisted_teacher_success: false,
    teacher_observations: 0, provider_calls: 0, training_admission: false, trace_admission: false,
  };
  await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  await writeFile(resolve(output, 'README.md'), `# ${shapeVersion.toUpperCase()} guided Neuralese soft-state source\n\nFresh scaffold-assisted source revision. It preserves the 12 authored source groups, splits, gold outputs, course AC-73 plus AC-73A clarification, and vaccine decision.json write boundary. The authored root instruction visibly supplies the typed eval scaffold without embedding per-world answers or notes.\n\nCurrent-pass notes preserve supported decision-relevant facts, distinguish historical events from the requested decision, and record uncertainty without inventing facts. The final interpreter applies exactly the explicit decisionRule to accumulated facts to determine the requested decision now; it does not require evidence that the decision was already executed or add unstated prerequisites. ${shapeVersion === 'v16' ? 'Known finite final fields use literal-union Draft types and explicit final enum contracts. The final prompt requires exact field formats and bare enum values without added unit labels.' : ''}\n\nThe scaffold calls a Neuralese<string> seed, four Neuralese<string> pass children, and one typed final Draft interpreter. Each pass opens only its current FileHandle.\n\nThe frozen V83 CPU proof is recorded separately. This is scaffold-assisted source graph validation, not unassisted teacher success, semantic truth, teacher observation, or training admission. No provider or teacher calls were made while building this artifact.\n\nSource SHA-256: ${sourceSha}.\n`);
  return { rows, sourceText, sourceSha, manifest };
}
