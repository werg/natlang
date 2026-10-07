import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here=dirname(fileURLToPath(import.meta.url));
const builder=resolve(here,'../scripts/inline-curriculum/build-semantic-source-worlds-v12.mjs');
const reviewedSha='33c2e405fbdda92f1f4f10ee0629c8cd96e36e170accd6a85b43cbca6ea9dabc';
const sha256=value=>createHash('sha256').update(value).digest('hex');
async function build(t){
  const parent=await mkdtemp(resolve(tmpdir(),'semantic-source-v12-'));
  t.after(()=>rm(parent,{recursive:true,force:true}));
  const out=resolve(parent,'fresh');
  const result=spawnSync(process.execPath,[builder,'--out',out],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  const text=await readFile(resolve(out,'source.cases.jsonl'),'utf8');
  const manifest=JSON.parse(await readFile(resolve(out,'source-manifest.json'),'utf8'));
  const review=JSON.parse(await readFile(resolve(out,'source-quality-review.json'),'utf8'));
  assert.equal(sha256(text),reviewedSha);
  assert.equal(manifest.source_cases_sha256,reviewedSha);
  return {out,text,manifest,review,rows:text.trimEnd().split('\n').map(JSON.parse)};
}

test('V12 builder reproduces the reviewed fresh cohort and refuses reuse',async t=>{
  const missing=spawnSync(process.execPath,[builder],{encoding:'utf8'});
  assert.notEqual(missing.status,0);assert.match(missing.stderr,/--out FRESH_DIRECTORY/);
  const {out,manifest,review,text,rows}=await build(t);
  assert.equal(rows.length,24);assert.equal(manifest.train_count,12);assert.equal(manifest.test_count,12);
  assert.equal(manifest.filehandle_count,12);assert.equal(manifest.iterate_count,12);
  assert.equal(review.checks.model_calls,0);assert.equal(review.checks.teacher_trajectories,0);assert.equal(review.checks.admission_granted,false);
  const again=spawnSync(process.execPath,[builder,'--out',out],{encoding:'utf8'});
  assert.notEqual(again.status,0);assert.equal(await readFile(resolve(out,'source.cases.jsonl'),'utf8'),text);
});

test('all V12 fact worlds and source groups are distinct and balanced by whole world',async t=>{
  const {rows,review}=await build(t);const groups=new Set(),worlds=new Set();
  for(const row of rows){
    assert.ok(!worlds.has(row.generation.independent_world));worlds.add(row.generation.independent_world);
    for(const group of row.source_groups){assert.ok(!groups.has(group),group);groups.add(group);}
    assert.deepEqual(row.source_groups,row.source_ids);assert.ok(row.source_groups.every(group=>group.startsWith('v12:')));
  }
  assert.equal(worlds.size,24);assert.equal(groups.size,84);
  assert.equal(rows.filter(x=>x.split==='train').length,12);assert.equal(rows.filter(x=>x.split==='test').length,12);
  assert.equal(review.checks.unique_source_groups,groups.size);assert.equal(review.checks.independent_worlds,worlds.size);
  const files=rows.filter(x=>x.curriculum.slice==='nested_scoped');
  assert.equal(files.length,12);
  const positiveCounts=files.map(row=>row.curriculum.reference.children.filter(c=>c.calls.at(-1)[1].value===true).length);
  assert.ok(positiveCounts.includes(1));assert.ok(positiveCounts.includes(2));
  for(const row of files){
    const task=JSON.parse(row.semantics.folder_files['task.json']);
    const code=row.curriculum.reference.root[0][1].code;
    assert.match(code,/nl\.with\(\{criterion,domain,window,decisionRule\}\)/);
    assert.match(code,/await judge\(file\)/);
    assert.match(code,/writeText\(JSON\.stringify\(ids\)\)/);
    assert.match(row.semantics.files['review_records.nl'],/same complete matching record FileHandle/);
    assert.ok(task.criterion&&task.domain&&task.window&&task.decision_rule);
    const children=row.curriculum.reference.children;
    assert.equal(children.length,4);
    const ids=Object.keys(row.semantics.folder_files).filter(x=>x.startsWith('records/')).map(x=>x.split('/').at(-1).replace('.md','')).sort();
    assert.deepEqual(children.map(x=>x.match.replace('.md','')).sort(),ids);
    const selected=children.filter(x=>x.calls.at(-1)[1].value===true).map(x=>x.match.replace('.md','')).sort();
    assert.deepEqual(row.semantics.expected,selected);
    for(const child of children){assert.equal(child.calls[0][1].path,`records/${child.match}`);assert.equal(typeof child.calls.at(-1)[1].value,'boolean');}
  }
});

test('iterateOn references make ordered current-pass-only repairs with exact string shape',async t=>{
  const {rows}=await build(t);const iter=rows.filter(x=>x.curriculum.slice==='iterate');
  assert.equal(iter.length,12);
  for(const row of iter){
    const task=JSON.parse(row.semantics.folder_files['task.json']);const fields=Object.keys(task.output_contract.fields);
    const outputType=`{ ${fields.map(k=>`${k}: string`).join('; ')} }`;
    const source=row.semantics.files['reconcile_item.nl'];const code=row.curriculum.reference.root[0][1].code;
    assert.ok(source.includes(`returns: ${JSON.stringify(outputType)}\n`),row.generation.independent_world);
    assert.ok(code.includes(`type Progress={pass:number;draft:${outputType}}`));
    assert.ok(code.includes(`Promise<${outputType}>`));assert.doesNotMatch(code,/Record<string,string>/);
    assert.match(code,/iterateOn\(revise/);assert.match(code,/withLimit\(\{maxSteps:task\.passes\.length\}\)/);
    assert.equal(task.passes.length,3);assert.equal(row.curriculum.reference.children.length,3);
    const children=row.curriculum.reference.children;let previous=task.initialDraft;const seen=new Set();
    for(let pass=0;pass<3;pass++){
      assert.equal(children[pass].calls[0][1].path,'packet.md');
      const current=children[pass].calls.at(-1)[1].value;
      assert.deepEqual(Object.keys(current),fields);assert.ok(Object.values(current).every(v=>typeof v==='string'));
      const changed=fields.filter(k=>current[k]!==previous[k]);assert.ok(changed.length>0,`${row.generation.independent_world} pass ${pass} makes no progress`);
      for(const key of changed){assert.ok(!seen.has(key),`${key} repaired before its own pass`);seen.add(key);assert.equal(current[key],row.semantics.expected[key]);}
      for(const key of fields)if(!seen.has(key))assert.equal(current[key],task.initialDraft[key],`${key} leaked from a later pass`);
      previous=current;
    }
    assert.deepEqual(previous,row.semantics.expected);assert.deepEqual([...seen].sort(),fields.slice().sort());
    const packet=row.semantics.folder_files['packet.md'];
    for(const [key,value] of Object.entries(row.semantics.expected))assert.ok(packet.includes(value)||packet.toUpperCase().includes(value.toUpperCase()),`${row.generation.independent_world}.${key} lacks source evidence`);
  }
});

test('numeric spellings and signed dispositions have visible task contracts',async t=>{
  const {rows}=await build(t);const by=Object.fromEntries(rows.map(x=>[x.generation.independent_world,x]));
  for(const slug of ['offshore_sling_inspection','vineyard_frost_response','turbine_blade_bond','special_collection_mold','rail_axle_ultrasonic','wildfire_radio_link','desalination_membrane_cip','shelter_generator_fuel','fiber_splice_loss','tanker_cip_rinse','geothermal_injection_sample','subsea_cable_insulation']){
    const row=by[slug],task=JSON.parse(row.semantics.folder_files['task.json']);
    assert.match(task.output_contract.number_copy_rule,/preserving sign and decimal places/);
    assert.equal(task.passes.at(-1).constraint.includes('signed'),true,`${slug} signature is not task-visible`);
  }
  assert.match(by.rail_switch_detection.semantics.folder_files['records/RS-201.md'],/signaling engineer E\. Park signed/);
  assert.match(by.wind_blade_bond.semantics.folder_files['records/WB-32.md'],/matching co-cured peel coupon passed/);
  assert.match(by.aquaculture_oxygen_calibration.semantics.folder_files['records/AO-52.md'],/expired 30 September 2026/);
  assert.match(by.bridge_bearing_grout.semantics.folder_files['records/BG-74.md'],/before the report was issued/);
  assert.match(by.carbon_capture_sample_chain.semantics.folder_files['records/CC-81.md'],/Seal 771 remained intact until lab receipt/);
});
