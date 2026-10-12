#!/usr/bin/env node
/** Convert the frozen paired-guidance requests to one native collector program per leaf call. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const sha256 = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}` : JSON.stringify(value);
const rows = text => text.trim().split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));

export function buildCollectorPrograms(requests, matchedPairs) {
  if (requests.length !== 48 || matchedPairs.length !== 24) throw new Error('expected 48 requests and 24 matched-pair bindings');
  const pairs = new Map(matchedPairs.map(pair => [pair.pair_id, pair]));
  if (pairs.size !== 24) throw new Error('matched-pair IDs must be unique');
  const seen = new Set();
  for (const request of requests) {
    const id = request.request_id;
    if (typeof id !== 'string' || seen.has(id)) throw new Error('request IDs must be unique strings');
    seen.add(id);
    const pair = pairs.get(request.pair_id);
    if (!pair || pair.case_id !== request.case_id || typeof pair.source_group !== 'string')
      throw new Error(`${id}: missing exact source-group binding`);
    if (!['A', 'B'].includes(request.arm) || !id.endsWith(`-${request.arm}`)) throw new Error(`${id}: invalid arm identity`);
    if (request.protocol !== 'One independent inline natural-language function call; return exactly one option string. No shared history between requests.')
      throw new Error(`${id}: unsupported protocol`);
    if (request.call?.result_type !== 'string' ||
        request.call?.signature !== 'nl.with<{ options: string[] }, string>({ options })`instruction`(state)')
      throw new Error(`${id}: unsupported typed call`);
    if (request.call?.capture_schema?.options !== 'string[]' || !Array.isArray(request.call?.captures?.options) ||
        request.call.captures.options.length !== 151 || new Set(request.call.captures.options).size !== 151)
      throw new Error(`${id}: expected the complete unique 151-option list`);
    if (typeof request.call?.input?.state !== 'string' || typeof request.call?.question_value !== 'string' ||
        typeof request.call?.instruction_template !== 'string' || !request.call.instruction_template.includes('${question}'))
      throw new Error(`${id}: missing leaf inputs`);
    if (Object.hasOwn(request, 'gold') || Object.hasOwn(request, 'expected') || Object.hasOwn(request.call, 'expected'))
      throw new Error(`${id}: model-facing input contains evaluator material`);
  }
  if (seen.size !== 48) throw new Error('expected 48 unique request IDs');
  for (const [pairId, pair] of pairs) {
    const arms = requests.filter(request => request.pair_id === pairId);
    if (arms.length !== 2 || new Set(arms.map(row => row.arm)).size !== 2)
      throw new Error(`${pairId}: expected exactly one request per arm`);
    const [a, b] = ['A', 'B'].map(arm => arms.find(row => row.arm === arm));
    for (const field of ['case_id', 'call.input.state', 'call.question_value', 'call.result_type']) {
      const get = row => field.split('.').reduce((value, key) => value?.[key], row);
      if (get(a) !== get(b)) throw new Error(`${pairId}: A/B mismatch in ${field}`);
    }
    if (sha256(canonical(a.call.captures.options)) !== sha256(canonical(b.call.captures.options)))
      throw new Error(`${pairId}: A/B options differ`);
  }

  return requests.map(request => {
    const sourceGroup = pairs.get(request.pair_id).source_group;
    const instruction = request.call.instruction_template.replaceAll('${question}', request.call.question_value);
    const rootCode = `const answer: string = await nl.with<{ options: string[] }, string>({ options })\`${instruction}\`(state);\nreturn answer;`;
    const rootSource = `---\nargs:\n  state: string\n  options: string[]\n  instruction: string\nreturns: string\n---\nReturn the typed decision produced by the fixed leaf call.\n`;
    return {
      version: 'natlang.program/2', id: `luna-guidance-ab-v2:${request.request_id}`, kind: 'lambda_source',
      source: 'paired-guidance-ab-v2-frozen-leaf-request', split: 'train', source_ids: [request.case_id],
      source_groups: [sourceGroup],
      protocol: { pair_id: request.pair_id, request_id: request.request_id, arm: request.arm,
        input_request_sha256: sha256(canonical(request)), training_admission: false },
      semantics: { root: 'decision.nl', files: { 'decision.nl': rootSource },
        inputs: { state: request.call.input.state, options: request.call.captures.options, instruction },
        expected: null, operation: 'exact' },
      curriculum: { trajectory_contract: { version: 'natlang.trajectory_contract/1', authored_root_eval: true,
        authored_root_sha256: sha256(rootCode), min_child_invocations: 1, min_state_edges_from_children: 0,
        min_iteration_steps: 0, required_child_roles: [{ role: 'paired_guidance_leaf', min_invocations: 1,
          instruction_contains: 'Answer from the scoped state.', return_type_contains: 'string' }] },
        reference: { root: [['eval', { code: rootCode }]] } },
    };
  });
}

async function main() {
  const artifactRoot = resolve(process.argv[2] ?? '/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1/provider-luna-clinc-full-option-guidance-v4/live-guided-v1/source-review');
  const source = join(artifactRoot, 'paired-guidance-ab-v2');
  const output = resolve(process.argv[3] ?? join(artifactRoot, 'paired-guidance-ab-v2-execution', 'prepared'));
  const [requestsText, pairsText, packageManifestText, packagePinText] = await Promise.all([
    readFile(join(source, 'requests.jsonl'), 'utf8'), readFile(join(source, 'matched-pairs.jsonl'), 'utf8'),
    readFile(join(source, 'manifest.json'), 'utf8'), readFile(join(source, 'manifest.sha256'), 'utf8')]);
  const packageManifest = JSON.parse(packageManifestText), packagePin = packagePinText.trim().split(/\s+/)[0];
  if (sha256(packageManifestText) !== packagePin) throw new Error('V2 package manifest SHA-256 mismatch');
  if (sha256(requestsText) !== packageManifest.files?.requests?.sha256 ||
      sha256(pairsText) !== packageManifest.files?.matched_pairs?.sha256)
    throw new Error('V2 request or matched-pair source digest mismatch');
  const programs = buildCollectorPrograms(rows(requestsText), rows(pairsText));
  const programText = programs.map(program => JSON.stringify(program)).join('\n') + '\n';
  await mkdir(output, { recursive: true });
  await writeFile(join(output, 'programs.jsonl'), programText);
  const builderText = await readFile(fileURLToPath(import.meta.url), 'utf8');
  const manifest = { schema: 'natlang.paired_guidance_ab_collector_ir/1',
    builder_script_sha256: sha256(builderText),
    source_package_manifest_sha256: packagePin, source_requests_sha256: sha256(requestsText),
    source_matched_pairs_sha256: sha256(pairsText),
    source_answerability_review_sha256: packageManifest.files.source_answerability_review.sha256,
    program_ir_sha256: sha256(programText), program_count: programs.length,
    execution: { provider_calls_made: false, default_action: 'prepare only',
      provider_path: 'scripts/teacher-collector.mjs + nativeJobRunner + Pi provider', execution_plans: false,
      planner: false,
      workers: 1, model_concurrency: 1, request_retries: 0, transport_retries: 0,
      output_capture: 'standard collector rows, provider evidence, trajectory and invocation traces',
      source_expected_labels_in_model_facing_ir: false, expected_value: null, training_admission: false,
      note: 'Each request is a fresh collector program. A SHA-bound authored root eval makes only the fixed typed nl.with child call; the child is the only sampled provider call. The exact oracle is intentionally unaccepted because expected is null; score outputs later with the separate evaluator key.' },
    cases: programs.map(program => ({ id: program.id, request_id: program.protocol.request_id,
      pair_id: program.protocol.pair_id, arm: program.protocol.arm, source_id: program.source_ids[0],
      source_group: program.source_groups[0], input_request_sha256: program.protocol.input_request_sha256,
      authored_root_eval_sha256: program.curriculum.trajectory_contract.authored_root_sha256 })) };
  const manifestText = JSON.stringify(manifest, null, 2) + '\n';
  await writeFile(join(output, 'runner-manifest.json'), manifestText);
  await writeFile(join(output, 'runner-manifest.sha256'), `${sha256(manifestText)}  runner-manifest.json\n`);
  console.log(JSON.stringify({ output: join(output, 'programs.jsonl'), program_count: programs.length,
    program_ir_sha256: sha256(programText), runner_manifest_sha256: sha256(manifestText), provider_calls_made: false }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
