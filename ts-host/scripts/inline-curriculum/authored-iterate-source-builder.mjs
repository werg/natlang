import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { curriculumCase } from './lib.mjs';

const canonical = value => JSON.stringify(value);

export function validateIterateWorlds(worlds) {
  if (!Array.isArray(worlds) || worlds.length !== 12) throw new Error('expected twelve independent iterate worlds');
  const slugs = new Set();
  const groups = new Set();
  for (const world of worlds) {
    if (!/^[a-z][a-z0-9_]+$/.test(world.slug) || slugs.has(world.slug)) throw new Error(`invalid or duplicate world slug: ${world.slug}`);
    slugs.add(world.slug);
    if (!world.group || groups.has(world.group)) throw new Error(`invalid or duplicate source group: ${world.group}`);
    groups.add(world.group);
    const fields = Object.keys(world.fields ?? {});
    if (fields.length < 4 || new Set(fields).size !== fields.length || fields.some(field => typeof world.initial?.[field] !== 'string'))
      throw new Error(`${world.slug}: expected at least four declared string fields and initial values`);
    if (world.passes?.length !== 4 || world.passStates?.length !== 4 || world.passes.some((pass, index) => !pass.name || !pass.evidence_path || !pass.constraint || !Array.isArray(pass.allowed_fields) || !world.passStates[index]))
      throw new Error(`${world.slug}: expected four scoped passes and four reference states`);
    const seen = new Set();
    let prior = world.initial;
    for (const [index, pass] of world.passes.entries()) {
      if (!pass.allowed_fields.length || pass.allowed_fields.some(field => !fields.includes(field)))
        throw new Error(`${world.slug}: pass ${index + 1} allows an undeclared/empty field set`);
      for (const field of pass.allowed_fields) seen.add(field);
      const next = world.passStates[index];
      if (fields.some(field => typeof next[field] !== 'string')) throw new Error(`${world.slug}: pass ${index + 1} is not a full string draft`);
      if (fields.some(field => !pass.allowed_fields.includes(field) && prior[field] !== next[field]))
        throw new Error(`${world.slug}: pass ${index + 1} changes a field outside its allowed set`);
      if (canonical(next) === canonical(prior)) throw new Error(`${world.slug}: pass ${index + 1} makes no change`);
      if (!world.evidence?.[pass.evidence_path]) throw new Error(`${world.slug}: missing evidence for ${pass.evidence_path}`);
      prior = next;
    }
    if (seen.size !== fields.length || fields.some(field => !seen.has(field)))
      throw new Error(`${world.slug}: pass fields do not cover the output shape`);
    if (prior !== world.passStates.at(-1)) throw new Error(`${world.slug}: final pass state mismatch`);
    if (!world.justified_revision?.reason || !world.justified_revision?.pass || !world.justified_revision?.field)
      throw new Error(`${world.slug}: missing revision provenance`);
    const { pass, field } = world.justified_revision;
    const beforeRevision = pass === 1 ? world.initial : world.passStates[pass - 2];
    if (pass < 2 || pass > 4 || !world.passes[pass - 1].allowed_fields.includes(field) ||
        world.passStates[pass - 1][field] === beforeRevision[field])
      throw new Error(`${world.slug}: declared justified revision is not an actual scoped field update`);
  }
}

function makeReferenceCode(world) {
  const fields = Object.keys(world.fields);
  const shape = `{ ${fields.map(field => `${field}: string`).join('; ')} }`;
  return `const task=await folder.file('task.json').readJson(); type Draft=${shape}; type Progress={pass:number;draft:Draft}; const revise=async (progress:Progress):Promise<Progress>=>{ const current=task.passes[progress.pass]; const evidence=await folder.file(current.evidence_path); const instruction=task.instruction; const contract=JSON.stringify(task.output_contract); const passName=current.name; const constraint=current.constraint; const allowedFields=JSON.stringify(current.allowed_fields); const currentDraft=JSON.stringify(progress.draft); const step:Neuralese<(source:FileHandle)=>Promise<Draft>>=nl.with({instruction,contract,passName,constraint,allowedFields,currentDraft})\`<|neuralese|>Read only this pass's complete source FileHandle, using the captured task instruction, output contract, current pass, constraint, allowed fields, and actual carried draft. Apply evidence from this pass. Update only allowed fields, preserving all other draft fields exactly. Compute any stated arithmetic, thresholds, event order, or authority condition from the source facts and task rule; do not copy a disposition unless the task asks for that signed disposition. Return the complete ${fields.length}-string draft exactly as declared.<|/neuralese|>\`; const next=await step(evidence); return {pass:progress.pass+1,draft:next}; }; const final=await iterateOn(revise,{pass:0,draft:task.initialDraft} as Progress).withLimit({maxSteps:task.passes.length}).until(p=>p.pass>=task.passes.length); await folder.file(task.output_path).writeText(JSON.stringify(final.draft)); const saved=await folder.file(task.output_path).readJson(); if(JSON.stringify(saved)!==JSON.stringify(final.draft)) throw Error('saved draft differs'); return final.draft;`;
}

export function makeIterateWorldCase(world, index, { family = 'authored_semantic_iterate_worlds_v15', familyVersion = 15, revision = 'iterate-semantic-worlds-v15/2' } = {}) {
  const fields = Object.keys(world.fields);
  const outputType = `{ ${fields.map(field => `${field}: string`).join('; ')} }`;
  const task = {
    instruction: world.instruction,
    output_path: 'decision.json',
    output_contract: {
      format: `JSON object with exactly these ${fields.length} string fields: ${fields.join(', ')}. No extra keys.`,
      fields: world.fields,
      carry_forward: 'Each pass may change only its allowed fields; preserve every other value in the actual carried draft exactly.',
      initial_state: 'initialDraft is an incomplete working placeholder; final output must satisfy all declared field formats and decision rules.',
      value_policy: world.value_policy,
      decision_rule: world.decision_rule,
    },
    initialDraft: world.initial,
    passes: world.passes.map(pass => ({
      name: pass.name,
      constraint: pass.constraint,
      allowed_fields: pass.allowed_fields,
      evidence_path: pass.evidence_path,
      source_scope: pass.source_scope,
    })),
  };
  const folderFiles = { 'task.json': canonical(task), 'decision.json': canonical(world.initial), ...world.evidence };
  const expected = world.passStates.at(-1);
  const expectedFiles = { ...folderFiles, 'decision.json': canonical(expected) };
  const code = makeReferenceCode(world);
  const children = world.passes.map((pass, passIndex) => ({
    match: `const passName: string = \\"${pass.name}\\";`,
    calls: [
      ['read_file', { path: pass.evidence_path }],
      ['return_result', { status: 'success', value: world.passStates[passIndex] }],
    ],
  }));
  const group = world.group;
  const record = curriculumCase({
    family, familyVersion,
    shape: `v15-${world.slug}-four-pass-derive-and-revise`,
    variant: 'evidence-scoped-iterative-derived-decision',
    splitGroup: group,
    split: index % 2 === 0 ? 'train' : 'test',
    slice: 'iterate',
    domain: world.domain,
    mode: 'single_call',
    inline: 'required',
    iterate: 'required',
    evidence: {
      world: [`Fictional ${world.domain} decision with source facts staged in four distinct records and a justified intermediate revision.`],
      retrieved: world.passes.map(pass => `One complete scoped FileHandle: ${pass.evidence_path}`),
      background: [],
    },
    assumptions: [],
    decisive: world.passes.map((pass, index) => ({ marker: pass.name, source: 'child', note: `Only pass ${index + 1} source and allowed fields apply; see source-quality-review.json for the derived update.` })),
    plausibleActions: ['apply only the current pass evidence and task rule to the carried draft'],
    minimumSequence: ['read the current pass task contract and exact source FileHandle', 'derive or revise only the allowed field values', 'carry the complete draft through four iterateOn steps', `write and read back the exact ${fields.length}-string result`],
    reference: {
      root: [['eval', { code }], ['return_result', { status: 'success', value: expected }]],
      children,
    },
    root: {
      name: 'reconcile_scoped_evidence', args: {}, returns: outputType, kind: 'directory-reducer',
      instructions: `Read task.json. Use iterateOn with typed Progress that carries the pass number and complete current draft. For each pass, pass only its named source FileHandle to a nested typed Neuralese function that captures the task instruction, full output contract, current pass and constraint, allowed fields, and JSON serialization of the actual carried draft. Apply the stated arithmetic and decision rules to the source facts; do not copy a final status from an unrelated source. Preserve every field outside this pass's allowed_fields. Write, read back, verify, and return exactly the ${fields.length} declared string fields: ${fields.join(', ')}.`,
    },
    inputs: {}, expected, folderFiles, expectedFiles,
  });
  record.source_ids = [group];
  record.source_groups = [group];
  record.source_revisions = [revision];
  record.generation = {
    generator: revision,
    independent_world: world.slug,
    capture_contract: {
      current_pass: 'task-visible exact pass name and constraint',
      allowed_fields: 'current-pass-specific, explicit list',
      current_draft: 'complete actual carried state serialized in closure',
      evidence: 'one exact current-pass FileHandle; future pass files are not passed to child',
    },
    source_quality: world.justified_revision.reason,
  };
  record.curriculum.explicit_fact_lineage = [group];
  return record;
}

export async function writeIterateCandidate({ out, worlds, revision = 'iterate-semantic-worlds-v15/2', qualityReview }) {
  validateIterateWorlds(worlds);
  const rows = worlds.map((world, index) => makeIterateWorldCase(world, index, { revision }));
  if (rows.length !== 12 || new Set(rows.map(row => row.source_groups[0])).size !== 12)
    throw new Error('source rows do not represent twelve independent factual groups');
  const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  const sourceSha = createHash('sha256').update(sourceText).digest('hex');
  const output = resolve(out);
  await mkdir(output, { recursive: false });
  await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
  const review = {
    schema: 'natlang.neuralese-source-quality-review/1',
    revision,
    source_cases_sha256: sourceSha,
    review_type: 'authored independent-world factual derivation and pass-local intermediate audit; independent root/reviewer assessment required',
    admission_granted: false,
    model_calls: 0,
    provider_calls: 0,
    teacher_trajectories: 0,
    counts: { task_variants: 12, factual_source_groups: 12, independent_factual_worlds: 12, train: 6, test: 6, passes: 48, justified_revisions: 12 },
    lineage: 'Twelve new factual worlds, each with its own four evidence files. No worlds or evidence reused from the V14 / V6r3 source family. These are source proposals, not teacher observations or admitted samples.',
    worlds: qualityReview,
  };
  await writeFile(resolve(output, 'source-quality-review.json'), JSON.stringify(review, null, 2) + '\n');
  const manifest = {
    schema: 'natlang.neuralese-semantic-iterate-v15/1', revision,
    source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
    world_count: 12, task_variant_count: 12, train_count: 6, test_count: 6, pass_count: 48,
    factual_source_groups: 12, quality_review: 'source-quality-review.json',
    scripted_proof: 'authored-scripted-proof/runtime-reference-proof.json',
    generation_status: 'candidate-only; root and independent semantic review pending',
    teacher_observations: 0, provider_calls: 0, training_admission: false, trace_admission: false,
    no_overlap_with: ['authored_semantic_source_worlds_v14', 'neuralese-v6r3-iterate-richness'],
  };
  await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  return { rows, sourceText, sourceSha, review, manifest };
}
