#!/usr/bin/env node
/** Freeze native authored programs and fixtures; do not generate assistant turns. */
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FIXTURE_ROOT, REPO, RELEASE, SOURCE_ROOT, json, jsonl, oracleLevel,
  readApplication, sha256 } from './lib.mjs';

export async function build({ repo = REPO, out }) {
  const sourceBytes = await readFile(join(repo, FIXTURE_ROOT, 'source-manifest.json'));
  const sourceIndex = JSON.parse(sourceBytes);
  const contractBytes = await readFile(join(repo, FIXTURE_ROOT, 'domain-contracts.json'));
  const contracts = JSON.parse(contractBytes);
  const operationalBytes = await readFile(join(repo, FIXTURE_ROOT, 'operational-contracts.json'));
  const operational = JSON.parse(operationalBytes);
  const names = (await readdir(join(repo, SOURCE_ROOT), { withFileTypes: true }))
    .filter(entry => entry.isDirectory() && /^\d{3}-/.test(entry.name)).map(entry => entry.name).sort();
  if (names.length !== 100 || names.some((name, index) => Number(name.slice(0, 3)) !== index + 1))
    throw new Error('all 100 distinct application IDs must be present before freezing');
  const applications = [];
  for (const name of names) {
    applications.push(await readApplication(repo, name, sourceIndex, contracts, operational));
    process.stderr.write(`validated ${applications.length}/${names.length}: ${name}\n`);
  }
  const rows = applications.flatMap(item => item.records);
  if (new Set(rows.map(row => row.id)).size !== rows.length) throw new Error('duplicate task ID');
  const groups = new Map();
  for (const row of rows) for (const group of row.source_groups) {
    if (groups.has(group) && groups.get(group) !== row.split) throw new Error(`cross-split group: ${group}`);
    groups.set(group, row.split);
  }
  await mkdir(resolve(out), { recursive: false });
  const outputs = [];
  for (const split of ['train', 'dev', 'test']) {
    const selected = rows.filter(row => row.split === split);
    for (const kind of ['gates', 'workflows']) {
      const partition = selected.filter(row => (oracleLevel(row.semantics.oracle) === 'exact') === (kind === 'gates'));
      const name = `${split}.${kind}.ir.jsonl`;
      const bytes = Buffer.from(jsonl(partition));
      await writeFile(join(out, name), bytes, { flag: 'wx' });
      outputs.push({ path: name, bytes: bytes.length, sha256: sha256(bytes), rows: partition.length, split,
        oracle: kind === 'gates' ? 'exact' : 'independent-review-required' });
    }
  }
  const inventory = applications.map(item => ({ application_id: String(item.app.application_id).padStart(3, '0'),
    title: item.app.title, domain_id: item.app.domain_id, split: item.records[0].split,
    source_root: `${SOURCE_ROOT}/${item.name}`, fixture_path: `${FIXTURE_ROOT}/${item.name}/cases.json`,
    brief_sha256: item.app.source_sha256, source_sha256: item.source_sha256,
    metadata_sha256: item.metadata_sha256, fixture_sha256: item.fixture_sha256,
    native_functions: item.native_functions, tasks: item.records.length,
    domain_contracts: item.domain_contracts,
    callable_scope: item.callable_scope,
    operational_coverage: item.operational_coverage,
    gate_tasks: item.records.filter(row => oracleLevel(row.semantics.oracle) === 'exact').length,
    workflow_tasks: item.records.filter(row => row.semantics.root === 'runCase.nl').length,
    stage_names: item.app.stages.map(stage => stage.name) }));
  const inventoryBytes = Buffer.from(json(inventory));
  await writeFile(join(out, 'inventory.json'), inventoryBytes, { flag: 'wx' });
  outputs.push({ path: 'inventory.json', bytes: inventoryBytes.length, sha256: sha256(inventoryBytes) });
  const manifest = {
    schema: 'natlang.design-brief-packet/1', release: RELEASE,
    source_manifest_sha256: sha256(sourceBytes), applications: applications.length,
    domain_contracts_sha256: sha256(contractBytes),
    operational_contracts_sha256: sha256(operationalBytes),
    domains: new Set(inventory.map(item => Number(item.domain_id))).size,
    domain_stages: applications.length * 5,
    native_functions: inventory.reduce((sum, item) => sum + item.native_functions, 0),
    tasks: rows.length, split_policy: 'whole-application: first-three-train-fourth-dev-fifth-test-in-each-domain',
    split_applications: Object.fromEntries(['train', 'dev', 'test'].map(split =>
      [split, inventory.filter(item => item.split === split).length])),
    files: outputs, training_admission: false,
    validation: { native_loader: true, typed_inputs: true, exact_expected_types: true,
      unique_ids: true, application_split_closure: true, original_brief_hashes: true,
      original_domain_fields_and_types: true,
      original_operational_details_and_workflow_mapping: true,
      view_durable_state_and_effect_proposal_assertions: true,
      named_helper_scope: true,
      compiler: 'run check.mjs separately', live_model: false, teacher_trajectories: 0, rl_rollouts: 0 },
  };
  await writeFile(join(out, 'manifest.json'), json(manifest), { flag: 'wx' });
  return manifest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length !== 2 || args[0] !== '--out') throw new Error('usage: node build.mjs --out NEW_DIRECTORY');
  const manifest = await build({ out: resolve(args[1]) });
  console.log(JSON.stringify({ output: resolve(args[1]), ...manifest }));
}
