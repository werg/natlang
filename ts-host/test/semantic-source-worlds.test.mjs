import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const builder = resolve(here, '../scripts/inline-curriculum/build-semantic-source-worlds-v10r2.mjs');
const reviewedSha = '9f26114f9bfc38766d78b81b7e313a465fbc5350a235b4f2a57d917e9c66e69d';
const sha256 = value => createHash('sha256').update(value).digest('hex');

async function generate(t) {
  const parent = await mkdtemp(resolve(tmpdir(), 'semantic-source-v10r2-'));
  t.after(() => rm(parent, { recursive: true, force: true }));
  const out = resolve(parent, 'fresh-output');
  const result = spawnSync(process.execPath, [builder, '--out', out], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const text = await readFile(resolve(out, 'source.cases.jsonl'), 'utf8');
  const manifest = JSON.parse(await readFile(resolve(out, 'source-manifest.json'), 'utf8'));
  assert.equal(sha256(text), reviewedSha);
  assert.equal(manifest.source_cases_sha256, reviewedSha);
  return { parent, out, text, manifest, rows: text.trimEnd().split('\n').map(JSON.parse) };
}

test('builder reproduces the reviewed V10r2 cases and refuses an existing output directory', async t => {
  const missingOutput = spawnSync(process.execPath, [builder], { encoding: 'utf8' });
  assert.notEqual(missingOutput.status, 0);
  assert.match(missingOutput.stderr, /--out FRESH_DIRECTORY/);
  const { out, text, manifest } = await generate(t);
  assert.equal(manifest.task_count, 24);
  assert.equal(manifest.train_count, 12);
  assert.equal(manifest.test_count, 12);
  const before = await readFile(resolve(out, 'source.cases.jsonl'), 'utf8');
  const again = spawnSync(process.execPath, [builder, '--out', out], { encoding: 'utf8' });
  assert.notEqual(again.status, 0);
  assert.equal(await readFile(resolve(out, 'source.cases.jsonl'), 'utf8'), before);
  assert.equal(text, before);
});

test('iterative root and child signatures exactly follow each visible output contract', async t => {
  const { rows } = await generate(t);
  const iterative = rows.filter(row => row.curriculum.slice === 'iterate');
  assert.equal(iterative.length, 12);
  for (const row of iterative) {
    const task = JSON.parse(row.semantics.folder_files['task.json']);
    const fields = Object.keys(task.output_contract.fields);
    const type = `{ ${fields.map(key => `${key}: string`).join('; ')} }`;
    const source = row.semantics.files['reconcile_item.nl'];
    const code = row.curriculum.reference.root[0][1].code;
    assert.ok(source.includes(`returns: ${JSON.stringify(type)}\n`), row.id);
    assert.match(source, /exactly these declared output_contract\.fields keys/);
    assert.ok(code.includes(`type Progress={pass:number;draft:${type}}`), row.id);
    assert.ok(code.includes(`Promise<${type}>`), row.id);
    assert.ok(!code.includes('Record<string,string>'), row.id);
    assert.equal(task.passes.length, 3);
    assert.match(task.output_contract.number_copy_rule, /preserve sign, decimal precision, exponent spelling and case/);
  }
});

test('source facts and all twelve iterative serialization rules are explicit', async t => {
  const { rows } = await generate(t);
  const bySlug = Object.fromEntries(rows.map(row => [row.generation.independent_world, row]));
  const positiveRecord = (slug, id) => bySlug[slug].semantics.folder_files[`records/${id}.md`].toLowerCase();
  assert.match(positiveRecord('soil_carbon_cores', 'SC-41'), /university freezer receipt/);
  assert.match(positiveRecord('seed_bank_accession', 'SB-810'), /cold vault/);
  assert.match(positiveRecord('battery_recycling', 'BR-301'), /receiving scale ticket net mass/);
  assert.match(positiveRecord('wildlife_camera_review', 'WC-17'), /authorized winter transect/);
  assert.match(positiveRecord('cleanroom_filter_change', 'CF-701'), /second person .* signed the torque witness/);

  const needs = {
    herbarium_relabel: { locality: ['curly quotation marks'], collector: ['exactly one ASCII space', 'field-number digits'] },
    bridge_bearing_repair: { bearing: ['one ASCII space on each side'], offset: ['one ASCII space and mm'] },
    microgrid_battery: { capacity: ['immediately after the number', 'no space'], release: ['exactly lowercase'] },
    archive_rights: { speaker: ['capitalization and internal spaces'], embargo: ['ASCII hyphens'], access: ['exactly lowercase'] },
    vaccine_cold_chain: { received: ['digits only', 'no comma or unit'], excursion: ['exceeded 8 °C', 'exactly lowercase'] },
    observatory_mirror: { reflectance: ['immediately after the token', 'no space'], coating: ['exact coating batch'] },
    rail_wheelset: { flange: ['one ASCII space', 'then mm'], release: ['exactly lowercase'] },
    museum_case_climate: { peak: ['one ASCII space', 'then °C'], disposition: ['exactly lowercase'] },
    orchard_irrigation: { pressure: ['one ASCII space', 'then kPa'], decision: ['exactly lowercase'] },
    lab_reagent_lot: { assay: ['immediately after the token', 'no space'], reagent: ['preserving spelling', 'internal spaces'] },
    coastal_tide_station: { benchmark: ['one ASCII space + rev + one ASCII space + revision letter'], residual: ['plus sign and trailing decimal zeros', 'one ASCII space and mm'] },
    water_meter_replacement: { address: ['unit, comma, and spaces'], flow: ['one ASCII space', 'then L/min'] },
  };
  for (const [slug, expected] of Object.entries(needs)) {
    const task = JSON.parse(bySlug[slug].semantics.folder_files['task.json']);
    for (const [field, fragments] of Object.entries(expected)) {
      const spec = task.output_contract.fields[field].toLowerCase();
      for (const fragment of fragments) assert.ok(spec.includes(fragment.toLowerCase()), `${slug}.${field} missing ${fragment}`);
    }
  }
  const herbarium = JSON.parse(bySlug.herbarium_relabel.semantics.folder_files['task.json']);
  assert.match(herbarium.passes[1].constraint, /full name \+ one ASCII space \+ field-number digits/i);
  const coast = JSON.parse(bySlug.coastal_tide_station.semantics.folder_files['task.json']);
  assert.match(coast.passes[1].constraint, /ID rev LETTER/);
});
