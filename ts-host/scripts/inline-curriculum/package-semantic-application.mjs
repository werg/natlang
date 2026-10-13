#!/usr/bin/env node
/** Snapshot an authored multi-file natlang application without rewriting its call graph. */
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { curriculumCase, returnCall } from './lib.mjs';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const args = process.argv.slice(2);
if (args.length !== 4 || args[0] !== '--manifest' || args[2] !== '--out')
  throw Error('usage: node package-semantic-application.mjs --manifest PROJECT_JSON --out FRESH_DIRECTORY');
const repo = resolve(import.meta.dirname, '../../..');
const manifestPath = resolve(args[1]);
const output = resolve(args[3]);
if (relative(repo, manifestPath).startsWith('..')) throw Error('manifest must be inside the repository');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const manifestBytes = await readFile(manifestPath);
const manifest = JSON.parse(manifestBytes);
if (!['natlang.semantic-application-source/1', 'natlang.application-semantic-project/1'].includes(manifest.schema))
  throw Error('unknown semantic application manifest schema');
if (typeof manifest.id !== 'string' || !/^[a-z][a-z0-9_-]{6,100}$/.test(manifest.id))
  throw Error('invalid project ID');

function safeRelative(path, label) {
  if (typeof path !== 'string' || !path || isAbsolute(path) || path.includes('\\') ||
      path.split('/').some(part => !part || part === '.' || part === '..'))
    throw Error(`unsafe ${label}: ${path}`);
  return path;
}
async function sourcePath(path, label) {
  const parts = safeRelative(path, label).split('/');
  let current = repo;
  for (const part of parts) {
    current = join(current, part);
    const stat = await lstat(current);
    if (stat.isSymbolicLink()) throw Error(`${label} contains a symlink: ${path}`);
  }
  if (relative(repo, current).startsWith('..')) throw Error(`${label} escapes repository: ${path}`);
  return current;
}
async function snapshot(rootPath, files, label, allowed) {
  if (!Array.isArray(files) || files.length === 0 || new Set(files).size !== files.length)
    throw Error(`${label} must list distinct files`);
  const result = {};
  for (const name of files) {
    safeRelative(name, label);
    if (!allowed(name)) throw Error(`${label} includes a nonruntime file: ${name}`);
    const file = await sourcePath(`${rootPath}/${name}`, label);
    const stat = await lstat(file);
    if (!stat.isFile()) throw Error(`${label} is not a regular file: ${name}`);
    result[name] = await readFile(file, 'utf8');
  }
  return result;
}
safeRelative(manifest.source_root, 'source_root');
safeRelative(manifest.fixture_root, 'fixture_root');
safeRelative(manifest.entry, 'entry');
const files = await snapshot(manifest.source_root, manifest.program_files, 'program_files',
  name => /\.(?:nl|ts|json)$/.test(name));
const fixture = await snapshot(manifest.fixture_root, manifest.fixture_files, 'fixture_files',
  () => true);
if (!Object.hasOwn(files, manifest.entry) || !manifest.entry.endsWith('.nl'))
  throw Error('entry must be an included .nl file');
const oraclePath = await sourcePath(manifest.oracle_path, 'oracle_path');
const requestPath = await sourcePath(manifest.request_path, 'request_path');
if (oraclePath.startsWith(resolve(repo, manifest.source_root) + '/') ||
    oraclePath.startsWith(resolve(repo, manifest.fixture_root) + '/') ||
    requestPath.startsWith(resolve(repo, manifest.fixture_root) + '/'))
  throw Error('request/oracle must remain outside the model-visible program and fixture trees');
const oracleBytes = await readFile(oraclePath);
const requestBytes = await readFile(requestPath);
const oracle = JSON.parse(oracleBytes);
const request = JSON.parse(requestBytes);
const rootSource = files[manifest.entry];
const frontmatter = rootSource.match(/^---\n([\s\S]*?)\n---\n/);
if (!frontmatter || !frontmatter[1].includes(`returns: ${manifest.result_type}`) ||
    !frontmatter[1].includes(`kind: ${manifest.entry_kind ?? 'directory-reducer'}`))
  throw Error('entry frontmatter does not match manifest result type/kind');
const inputBindings = manifest.input_bindings ?? { request: { from: 'request_path' } };
const inputs = {};
const inputSourceHashes = { request_path: sha(requestBytes) };
for (const [name, binding] of Object.entries(inputBindings)) {
  if (binding?.from === 'request_path') inputs[name] = request;
  else if (binding?.from === 'snapshot_path') {
    const snapshotPath = await sourcePath(manifest.snapshot_path, 'snapshot_path');
    const bytes = await readFile(snapshotPath);
    inputSourceHashes.snapshot_path = sha(bytes);
    inputs[name] = JSON.parse(bytes);
  } else throw Error(`unknown input binding for ${name}`);
}
if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs))
  throw Error('input_bindings must be an object');
for (const key of Object.keys(inputs))
  if (!new RegExp(`^\\s+${key}:`, 'm').test(frontmatter[1]))
    throw Error(`entry has no declared input ${key}`);
const graph = manifest.typed_call_graph;
if (graph?.nodes && graph?.edges) {
  const nodes = new Set(graph.nodes.map(node => node.path));
  if (!nodes.has(manifest.entry) || [...nodes].some(path => !Object.hasOwn(files, path) || !path.endsWith('.nl')) ||
      Object.keys(files).filter(path => path.endsWith('.nl')).some(path => !nodes.has(path)))
    throw Error('typed_call_graph nodes differ from packaged .nl files');
  for (const edge of graph.edges) {
    const folder = edge.caller.replace(/\.nl$/, '');
    const declaredUses = files[edge.caller]?.match(/^uses:\s*\[([^\]]*)\]/m)?.[1]
      .split(',').map(value => value.trim()) ?? [];
    const explicitUse = declaredUses.includes(edge.callee.replace(/\.nl$/, ''));
    if (!nodes.has(edge.caller) || !nodes.has(edge.callee) ||
        (dirname(edge.callee) !== folder && !explicitUse))
      throw Error(`noncallable edge: ${edge.caller} -> ${edge.callee}`);
  }
}
const visible = JSON.stringify({ files, fixture, inputs });
const fingerprint = sha(Buffer.from(visible));
const nlSources = Object.entries(files).filter(([path]) => path.endsWith('.nl'));
const numberedPolicySteps = nlSources.reduce((count, [, body]) => count +
  body.split('\n').filter(line => /^\s*\d+\.\s+\S/.test(line)).length, 0);
const scopedSites = manifest.scoped_literals ?? manifest.scoped_literal_sites ?? [];
const scopedSitePaths = scopedSites.map(site => typeof site === 'string' ? site : site.site);
const graphEdges = graph?.edges ?? [];
const distance = new Map([[manifest.entry, 0]]);
let graphCycleSuspected = false;
for (let pass = 0; pass < nlSources.length; pass++) {
  let changed = false;
  for (const edge of graphEdges) if (distance.has(edge.caller) &&
      (distance.get(edge.callee) ?? -1) < distance.get(edge.caller) + 1) {
    distance.set(edge.callee, distance.get(edge.caller) + 1);
    changed = true;
  }
  if (!changed) break;
  if (pass === nlSources.length - 1) graphCycleSuspected = true;
}
const qualityAudit = { schema: 'natlang.semantic-application-quality-observations/1', gating: false,
  nl_lines: nlSources.reduce((count, [, body]) => count + body.split('\n').length, 0),
  numbered_policy_steps: numberedPolicySteps,
  callable_nodes: nlSources.length, call_graph_edges: graphEdges.length,
  declared_max_call_depth: graphCycleSuspected ? null : Math.max(...distance.values()),
  graph_cycle_suspected: graphCycleSuspected,
  declared_scoped_literal_sites: scopedSitePaths.length,
  sites_with_nl_instruction: scopedSitePaths.filter(path => typeof files[path] === 'string' &&
    /\bnl(?:\.with)?\b/i.test(files[path])).length,
  declared_literal_consumers: scopedSites.filter(site => typeof site === 'object' &&
    typeof site.consumed_by === 'string' && site.consumed_by.trim()).length,
  crisp_policy_displacement: 'manual_review_pending',
  live_literal_consumption: 'unmeasured',
  note: 'These are static observations and authored claims; live semantic work and policy location require trace review. Scale and depth are mix labels, never admission limits.' };
const hasFullGold = Object.hasOwn(oracle, 'expected') &&
  manifest.comparison_spec?.schema_version === 'semantic-structured-comparison/1';
const rootName = manifest.entry.replace(/\.nl$/, '');
const row = curriculumCase({ family: 'semantic_application_project', shape: manifest.id,
  variant: fingerprint.slice(0, 12), splitGroup: manifest.source_world_split_key ?? manifest.id,
  slice: 'iterate', domain: manifest.domain ?? 'other', mode: 'single_call',
  inline: 'required', iterate: 'required', reference: [],
  root: { name: rootName, args: {}, returns: manifest.result_type, kind: manifest.entry_kind ?? 'directory-reducer',
    instructions: 'Read the authored natlang application source.' }, files: {}, inputs,
  expected: hasFullGold ? oracle.expected : null, folderFiles: fixture, expectedFiles: fixture });
row.id = `semantic-application:${manifest.id}`;
row.source = 'natlang-authored-application';
row.source_ids = [row.id];
row.source_groups = [manifest.source_world_split_key ?? manifest.id];
row.source_revisions = [`sha256:${fingerprint}`];
row.license = manifest.provenance?.license ?? manifest.license_provenance ?? 'repository-authored';
row.gold_sources = ['authored-application-oracle'];
row.semantics.root = manifest.entry;
row.semantics.files = files;
row.semantics.inputs = inputs;
row.semantics.folder_files = fixture;
row.semantics.expected_files = fixture;
if (Array.isArray(manifest.services)) {
  row.semantics.services = {};
  for (const service of manifest.services) {
    if (typeof service?.binding !== 'string' || !/^[A-Za-z][A-Za-z0-9_]*$/.test(service.binding) ||
        !Object.hasOwn(files, service.declaration) ||
        Object.hasOwn(row.semantics.services, service.binding))
      throw Error('invalid or duplicate external service declaration');
    const hostPath = await sourcePath(service.host_implementation, 'service host implementation');
    row.semantics.services[service.binding] = await readFile(hostPath, 'utf8');
  }
}
if (hasFullGold) row.semantics.oracle = { level: 'normalized', normalization: 'semantic-structured',
  comparison_spec: manifest.comparison_spec };
let exactVerifierHash = null;
if (manifest.external_verifier_path) {
  const verifierPath = await sourcePath(manifest.external_verifier_path, 'external_verifier_path');
  if (!manifest.external_verifier_path.startsWith('training/semantic_applications/') ||
      !manifest.external_verifier_path.endsWith('.mjs') ||
      verifierPath.startsWith(resolve(repo, manifest.source_root) + '/') ||
      verifierPath.startsWith(resolve(repo, manifest.fixture_root) + '/'))
    throw Error('external verifier must be a separate training/semantic_applications .mjs file');
  exactVerifierHash = sha(await readFile(verifierPath));
  row.semantics.trusted_exact_verifier = { path: manifest.external_verifier_path,
    sha256: exactVerifierHash, oracle };
}
row.generation.collection = 'teacher';
row.generation.source_quality = 'Exact authored natlang project snapshot; external verifier and independent semantic review required.';
row.collection_guidance = { training_admission: false, review_scope: 'full-project-trajectory-and-external-oracle' };
await mkdir(output, { recursive: false });
await writeFile(join(output, 'source-package.ir.jsonl'), JSON.stringify(row) + '\n');
let canonicalReferenceReplay = null;
if (manifest.reference_path) {
  const referencePath = await sourcePath(manifest.reference_path, 'reference_path');
  if (referencePath.startsWith(resolve(repo, manifest.source_root) + '/') ||
      referencePath.startsWith(resolve(repo, manifest.fixture_root) + '/'))
    throw Error('reference must remain outside model-visible trees');
  const referenceBytes = await readFile(referencePath);
  const reference = JSON.parse(referenceBytes);
  const replayOnly = structuredClone(row);
  replayOnly.semantics.expected = reference;
  replayOnly.semantics.oracle = 'exact';
  delete replayOnly.semantics.trusted_exact_verifier;
  // A canonical typed witness can prove source/result plumbing, not that the
  // actual NL application used its scoped literals or iterateOn. Those remain
  // live-trajectory obligations, so remove technique checks only on this
  // separately named replay artifact.
  replayOnly.curriculum.inline = 'optional';
  delete replayOnly.curriculum.iterate;
  replayOnly.curriculum.reference = { root: [returnCall(reference)], children: [] };
  const [proof] = await verifyCases([replayOnly], TOOLS_PROMPT);
  canonicalReferenceReplay = { path: manifest.reference_path, sha256: sha(referenceBytes),
    reference_ok: proof.ok, reference_problems: proof.problems,
    opening_sha256: proof.opening_sha256,
    scope: 'source/type replay of one canonical witness, not complete semantic grading or teacher admission' };
  await writeFile(join(output, 'reference-replay-only.ir.jsonl'), JSON.stringify(replayOnly) + '\n');
}
const report = { schema: 'natlang.semantic-application-package/1', id: manifest.id,
  source_status: hasFullGold ? 'candidate_pending_reference_and_semantic_review' :
    exactVerifierHash ? 'held_exact_verifier_bound_semantic_review_pending' :
      'held_invariant_oracle_requires_external_verifier',
  training_admission: false, program_file_count: Object.keys(files).length,
  nl_file_count: Object.keys(files).filter(path => path.endsWith('.nl')).length,
  fixture_file_count: Object.keys(fixture).length, graph_edge_count: graph?.edges?.length ?? null,
  manifest: relative(repo, manifestPath), manifest_sha256: sha(manifestBytes),
  oracle_sha256: sha(oracleBytes), input_source_hashes: inputSourceHashes,
  service_bindings: Object.keys(row.semantics.services ?? {}),
  visible_sha256: fingerprint, source_sha256: sha(Buffer.from(JSON.stringify(row) + '\n')),
  external_verifier_path: manifest.external_verifier_path ?? null,
  external_verifier_sha256: exactVerifierHash,
  canonical_reference_replay: canonicalReferenceReplay,
  quality_audit: qualityAudit,
  note: 'Program source preserves authored bytes and callable paths. Packaging does not admit or execute the case.' };
await writeFile(join(output, 'package-report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, ...report }));
