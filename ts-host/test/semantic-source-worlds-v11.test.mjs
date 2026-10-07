import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here=dirname(fileURLToPath(import.meta.url));
const builder=resolve(here,'../scripts/inline-curriculum/build-semantic-source-worlds-v11.mjs');
const reviewedSha='26c5b1fa8bf0b48f7da63742251288c08eeab6dedfe74265c62591fe1bbedd45';
const sha256=value=>createHash('sha256').update(value).digest('hex');
async function build(t){
  const parent=await mkdtemp(resolve(tmpdir(),'semantic-source-v11-'));
  t.after(()=>rm(parent,{recursive:true,force:true}));
  const out=resolve(parent,'fresh');
  const result=spawnSync(process.execPath,[builder,'--out',out],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  const text=await readFile(resolve(out,'source.cases.jsonl'),'utf8');
  const manifest=JSON.parse(await readFile(resolve(out,'source-manifest.json'),'utf8'));
  assert.equal(sha256(text),reviewedSha);
  assert.equal(manifest.source_cases_sha256,reviewedSha);
  return {out,text,manifest,rows:text.trimEnd().split('\n').map(JSON.parse)};
}

test('V11 builder reproduces the reviewed 24-case cohort and protects existing outputs',async t=>{
  const missing=spawnSync(process.execPath,[builder],{encoding:'utf8'});
  assert.notEqual(missing.status,0);
  assert.match(missing.stderr,/--out FRESH_DIRECTORY/);
  const {out,manifest,text,rows}=await build(t);
  assert.equal(rows.length,24);
  assert.equal(manifest.train_count,12);
  assert.equal(manifest.test_count,12);
  assert.equal(manifest.filehandle_count,12);
  assert.equal(manifest.iterate_count,12);
  assert.doesNotMatch(text,/(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])/i,'digest claims must not use truncated 32-hex values');
  const again=spawnSync(process.execPath,[builder,'--out',out],{encoding:'utf8'});
  assert.notEqual(again.status,0);
  assert.equal(await readFile(resolve(out,'source.cases.jsonl'),'utf8'),text);
});

test('V11 has independent grouped splits and scope-complete FileHandle reducer contracts',async t=>{
  const {rows}=await build(t);
  const groups=new Map();
  for(const row of rows){
    for(const group of row.source_groups){assert.ok(!groups.has(group),group);groups.set(group,row.split);}
    assert.equal(row.source_groups.length,row.source_ids.length);
    assert.equal(row.generation.independent_world.length>0,true);
  }
  const fileRows=rows.filter(row=>row.curriculum.slice==='nested_scoped');
  assert.equal(fileRows.length,12);
  for(const row of fileRows){
    const code=row.curriculum.reference.root[0][1].code;
    const root=row.semantics.files['review_records.nl'];
    assert.match(root,/criterion/);assert.match(root,/domain/);assert.match(root,/window/);assert.match(root,/decision rule/);
    assert.match(code,/nl\.with\(\{criterion,domain,window,decisionRule\}\)/);
    const task=JSON.parse(row.semantics.folder_files['task.json']);
    assert.ok(task.criterion&&task.domain&&task.window&&task.decision_rule);
    const children=row.curriculum.reference.children;
    assert.equal(children.length,4);
    assert.equal(new Set(children.map(child=>child.match)).size,4);
    assert.deepEqual(children.map(child=>child.match).sort(),Object.keys(row.semantics.folder_files).filter(path=>path.startsWith('records/')).map(path=>path.split('/').pop()).sort());
    assert.equal(children.filter(child=>child.calls.at(-1)[1].value===true).length,1,row.generation.independent_world);
  }
});

test('all iterative worlds use exact task field shapes and current-pass-only carried drafts',async t=>{
  const {rows}=await build(t);
  const iters=rows.filter(row=>row.curriculum.slice==='iterate');
  assert.equal(iters.length,12);
  for(const row of iters){
    const task=JSON.parse(row.semantics.folder_files['task.json']);
    const fields=Object.keys(task.output_contract.fields);
    const type=`{ ${fields.map(key=>`${key}: string`).join('; ')} }`;
    const nl=row.semantics.files['reconcile_item.nl'];
    const code=row.curriculum.reference.root[0][1].code;
    assert.ok(nl.includes(`returns: ${JSON.stringify(type)}\n`),row.generation.independent_world);
    assert.match(nl,/exactly the declared output_contract\.fields keys/);
    assert.ok(code.includes(`type Progress={pass:number;draft:${type}}`));
    assert.ok(code.includes(`Promise<${type}>`));
    assert.ok(!code.includes('Record<string,string>'));
    assert.equal(task.passes.length,3);
    const children=row.curriculum.reference.children;
    assert.equal(children.length,3);
    const initial=task.initialDraft;
    let previous=initial;
    const firstTargetPass={};
    for(let pass=0;pass<3;pass++){
      const current=children[pass].calls.at(-1)[1].value;
      assert.deepEqual(Object.keys(current),fields);
      assert.ok(Object.values(current).every(value=>typeof value==='string'));
      const changed=fields.filter(key=>current[key]!==previous[key]);
      assert.ok(changed.length>0,`${row.generation.independent_world} pass ${pass} makes no progress`);
      for(const key of changed)assert.equal(firstTargetPass[key],undefined,`${key} revised before`),firstTargetPass[key]=pass;
      for(const key of fields){
        if(firstTargetPass[key]===undefined)assert.equal(current[key],initial[key],`${key} leaked from a later pass`);
        else assert.equal(current[key],row.semantics.expected[key]);
      }
      previous=current;
    }
    assert.deepEqual(previous,row.semantics.expected);
    assert.deepEqual(Object.keys(row.semantics.expected),fields);
    assert.match(task.output_contract.number_copy_rule,/preserve sign, decimal precision, exponent spelling and case/);
    const packet=row.semantics.folder_files['packet.md'];
    for(const [field,value] of Object.entries(row.semantics.expected))
      assert.ok(packet.includes(value)||packet.includes(value.toUpperCase()),`${row.generation.independent_world}.${field} gold is not stated in its packet`);
  }
});

test('high-risk source evidence and visible serialization rules stay present',async t=>{
  const {rows}=await build(t);
  const bySlug=Object.fromEntries(rows.map(row=>[row.generation.independent_world,row]));
  const rec=(slug,id)=>bySlug[slug].semantics.folder_files[`records/${id}.md`];
  assert.match(rec('food_allergen_lot','FA-41'),/verified 24-minute flush/);
  assert.match(rec('forensic_disk_imaging','FD-71'),/SHA-256 are both [0-9a-f]{64}/);
  assert.match(rec('traffic_signal_recommission','TS-88'),/firmware SHA-256 matches release manifest [0-9a-f]{64}/);
  assert.match(rec('crane_lift_permit','CL-18'),/Licensed signaler/);
  assert.match(rec('seismic_station_qc','SQ-31'),/current revision/);
  assert.match(rec('clinical_analyzer_qc','CQ-14'),/clinical laboratory scientist/i);
  assert.match(rec('protective_relay_test','PR-44'),/secondary-injection test/);
  const trees=bySlug.urban_tree_record;
  const treeTask=JSON.parse(trees.semantics.folder_files['task.json']);
  assert.match(treeTask.window,/15 June 2026/);
  for(const id of ['UT-206','UT-207','UT-208','UT-209']) assert.match(trees.semantics.folder_files[`records/${id}.md`],/15 June 2026/);
  assert.match(rec('urban_tree_record','UT-206'),/valid through 31 December 2026/);
  assert.match(rec('urban_tree_record','UT-208'),/expired 31 March 2026/);
  const qc=bySlug.clinical_analyzer_qc;
  assert.match(JSON.parse(qc.semantics.folder_files['task.json']).window,/20 June 2026 at 08:20/);
  assert.match(rec('clinical_analyzer_qc','CQ-14'),/20 June 2026 at 07:30.*20 June 2026 at 08:05.*08:20 that day/);
  assert.match(rec('clinical_analyzer_qc','CQ-15'),/19 June 2026 at 06:00.*20 June 2026 at 08:20, 26 hours 20 minutes later/);
  const signed = [
    ['grain_silo_clearance','Safety officer signs PROHIBITED'],
    ['borehole_water_log','Geotechnical lead signs RESURVEY'],
    ['ev_charger_commission','Electrical inspector signs REVISIT'],
    ['ferry_raft_service','Marine inspector signs SERVICE'],
    ['operating_room_suction','Biomedical engineer signs REPAIR'],
    ['floodgate_drive','Dam safety engineer signs REPLACE'],
    ['digital_archive_fixity','archivist signs RESTRICTED'],
    ['city_bus_brakes','Fleet engineer signs HOLD'],
    ['avalanche_sensor','Mountain-safety lead signs INSPECT'],
    ['groundwater_isotope','Hydrogeologist signs RESAMPLE'],
  ];
  for(const [slug,phrase] of signed) assert.ok(bySlug[slug].semantics.folder_files['packet.md'].includes(phrase),`${slug} missing explicit signed disposition`);
  const archive=JSON.parse(bySlug.digital_archive_fixity.semantics.folder_files['task.json']);
  assert.match(archive.output_contract.fields.digest,/complete lowercase SHA-256 digest exactly/);
  assert.equal(bySlug.digital_archive_fixity.semantics.expected.digest.length,64);
  for(const [slug,field,pattern] of [
    ['runway_beacon','intensity',/^\d+(?:\.\d+)? cd$/],
    ['grain_silo_clearance','phosphine',/^\d+(?:\.\d+)? ppm$/],
    ['borehole_water_log','depth',/^\d+(?:\.\d+)? m below datum$/],
    ['ev_charger_commission','output',/^\d+(?:\.\d+)? kW$/],
    ['emergency_repeater','rssi',/^−\d+\.\d{2} dBm$/],
    ['ferry_raft_service','cylinder',/^\d+\.\d{2} kg$/],
    ['operating_room_suction','flow',/^\d+(?:\.\d+)? L\/min$/],
    ['floodgate_drive','torque',/^\d+(?:\.\d+)? N·m$/],
    ['city_bus_brakes','efficiency',/^\d+\.\d{2}%$/],
    ['avalanche_sensor','minimum',/^−\d+\.\d{2} °C$/],
    ['groundwater_isotope','delta18o',/^−\d+\.\d{3}‰$/],
  ]) assert.match(bySlug[slug].semantics.expected[field],pattern,`${slug}.${field} literal format`);
  const isotope=JSON.parse(bySlug.groundwater_isotope.semantics.folder_files['task.json']);
  assert.match(isotope.output_contract.fields.delta18o,/signed δ18O token/);
  assert.match(isotope.output_contract.fields.delta18o,/‰ immediately/);
  assert.match(isotope.output_contract.fields.delta18o,/no space/);
});
