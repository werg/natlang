#!/usr/bin/env node
/** Compare independent fixture truth to the original PR; summarize source architecture without claiming execution. */
import { execFileSync } from 'node:child_process';
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { FIXTURE_ROOT, SOURCE_ROOT, REPO, RELEASE, json, sha256 } from './lib.mjs';

const args = process.argv.slice(2);
if (args.length !== 6 || args[0] !== '--baseline' || args[2] !== '--packet' || args[4] !== '--out')
  throw new Error('usage: node audit.mjs --baseline ORIGINAL_COMMIT --packet FROZEN_DIRECTORY --out REPORT_DIRECTORY');
const baseline = args[1], packet = resolve(args[3]), out = resolve(args[5]);
if (!/^[a-f0-9]{40}$/.test(baseline)) throw new Error('baseline must be an exact commit identity');
const canonical = value => JSON.stringify(value, (_, x) => x && typeof x === 'object' && !Array.isArray(x)
  ? Object.fromEntries(Object.keys(x).sort().map(k => [k, x[k]])) : x);
function objects(value, all = []) {
  if (value && typeof value === 'object') {
    if (!Array.isArray(value)) all.push(value);
    for (const item of Object.values(value)) objects(item, all);
  }
  return all;
}
function includesRecord(candidates, record) {
  return candidates.some(candidate => Object.keys(record).every(key =>
    Object.hasOwn(candidate, key) && canonical(candidate[key]) === canonical(record[key])));
}
function values(value, all = new Set()) {
  if (value && typeof value === 'object') for (const child of Object.values(value)) values(child, all);
  else if (typeof value === 'string') all.add(value);
  return all;
}
const names = (await readdir(join(REPO, SOURCE_ROOT))).filter(name => /^\d{3}-/.test(name)).sort();
const fixtureResults = [];
for (const name of names) {
  const path = `${FIXTURE_ROOT}/${name}/cases.json`;
  const oldBytes = execFileSync('git', ['show', `${baseline}:${path}`], { cwd: REPO, maxBuffer: 5 * 1024 * 1024 });
  const newBytes = await readFile(join(REPO, path));
  const previous = JSON.parse(oldBytes), current = JSON.parse(newBytes);
  const currentById = new Map(current.map(item => [item.id, item]));
  const issues = [], details = [];
  if (current.length !== previous.length || currentById.size !== previous.length) issues.push('case identity/count changed');
  for (const old of previous) {
    const now = currentById.get(old.id), errors = [];
    if (!now) { issues.push(`missing ${old.id}`); continue; }
    for (const [key, value] of Object.entries(old)) if (!['root', 'inputs'].includes(key) && canonical(now[key]) !== canonical(value))
      errors.push(`independent ${key} changed`);
    const exact = old.oracle === 'exact' || old.oracle?.level === 'exact';
    if (exact && canonical(old.inputs) !== canonical(now.inputs)) errors.push('exact qualification input changed');
    if (!exact) {
      const supplied = objects(now.inputs), oldSnapshot = old.inputs.snapshot ?? {};
      const sourceRecords = [...(oldSnapshot.caseRecords ?? []), ...(oldSnapshot.policyCandidates ?? []),
        ...(oldSnapshot.evidencePages ?? []).flatMap(page => page.evidence ?? [])];
      for (const record of sourceRecords) if (!includesRecord(supplied, record)) errors.push(`source record lost or changed: ${record.id ?? record.source ?? 'anonymous'}`);
      const currentValues = values(now.inputs);
      for (const value of values(old.inputs)) if (!currentValues.has(value)) errors.push(`source value lost: ${value.slice(0,100)}`);
    }
    details.push({ id: old.id, kind: exact ? 'exact' : 'judged', original_root: old.root, current_root: now.root,
      independent_truth_sha256: sha256(canonical({ expected: old.expected, oracle: old.oracle, boundary_assertions: old.boundary_assertions })),
      preserved: errors.length === 0, ...(errors.length ? { issues: errors } : {}) });
    issues.push(...errors.map(error => `${old.id}: ${error}`));
  }
  fixtureResults.push({ application: name, before_sha256: sha256(oldBytes), after_sha256: sha256(newBytes),
    cases: details.length, preserved: issues.length === 0, details, ...(issues.length ? { issues } : {}) });
}
await mkdir(out, { recursive: true });
const migration = { schema: 'natlang.design-brief-fixture-migration/1', release: RELEASE, baseline_commit: baseline,
  evidence: 'Exact comparisons of independent expectations/oracles, exact gate inputs, original source records and source scalar values; no model execution.',
  applications: fixtureResults.length, tasks: fixtureResults.reduce((n,x) => n + x.cases,0),
  all_preserved: fixtureResults.length === 100 && fixtureResults.every(item => item.preserved), results: fixtureResults };
await writeFile(join(out, 'fixture-migration.json'), json(migration));
const inventory = JSON.parse(await readFile(join(packet, 'inventory.json'), 'utf8'));
const helperGroups = new Map(), shapeGroups = new Map(), architecture = [];
for (const item of inventory) {
  const name = names.find(name => name.startsWith(item.application_id + '-'));
  const manifestBytes = await readFile(join(REPO, SOURCE_ROOT, name, 'application.json'));
  const manifest = JSON.parse(manifestBytes), graph = item.potential_calls;
  for (const path of manifest.source_files.filter(path => path.endsWith('.nl'))) {
    const leaf = path.split('/').at(-1);
    const group = helperGroups.get(leaf) ?? new Set(); group.add(item.application_id); helperGroups.set(leaf, group);
  }
  const adjacency = new Map();
  for (const { caller, callee } of graph.edges) {
    const children = adjacency.get(caller) ?? []; children.push(callee); adjacency.set(caller, children);
  }
  function shape(path, ancestry = new Set()) {
    if (ancestry.has(path)) return 'cycle';
    return '(' + (adjacency.get(path) ?? []).map(child => shape(child, new Set([...ancestry, path]))).sort().join('') + ')';
  }
  const signature = manifest.interfaces.map(iface => shape(iface.path)).sort().join('|');
  const group = shapeGroups.get(signature) ?? []; group.push(item.application_id); shapeGroups.set(signature, group);
  architecture.push({ application_id: item.application_id, title: item.title, source_sha256: item.source_sha256,
    interface_sha256: sha256(manifestBytes), public_operations: manifest.interfaces.map(iface => ({ name: iface.name, path: iface.path, purpose: iface.purpose })),
    native_functions: item.native_functions, potential_edges: graph.edges.length, unreferenced: graph.unreferenced,
    source_topology_fingerprint: sha256(signature), decomposition: `${SOURCE_ROOT}/${name}/DECOMPOSITION.md` });
}
const review = { schema: 'natlang.design-brief-architecture-review/1', release: RELEASE,
  evidence: 'Source potential-call topology and public contracts, not observed calls, measured model behavior, or a proof of semantic uniqueness.',
  applications: architecture.length, public_operations: architecture.reduce((n,x) => n+x.public_operations.length,0),
  comparison_policy: 'A shared structural fingerprint is a review prompt, not a failure or a diversity score. Distinct names and fingerprints cannot establish independent semantic approaches.',
  repeated_source_topologies: [...shapeGroups.values()].filter(group => group.length > 1),
  reused_helper_names: [...helperGroups].filter(([, group]) => group.size > 1).map(([name, group]) => ({ name, applications: [...group] })),
  results: architecture };
await writeFile(join(out, 'architecture.json'), json(review));
console.log(json({ applications: migration.applications, tasks: migration.tasks, all_preserved: migration.all_preserved,
  public_operations: review.public_operations, shared_topology_groups: review.repeated_source_topologies.length,
  failures: fixtureResults.filter(item => !item.preserved).map(item => ({ application:item.application, issues:item.issues })) }));
process.exitCode = migration.all_preserved ? 0 : 1;
