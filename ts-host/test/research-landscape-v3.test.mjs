import test from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
import { buildResearchLandscapeV3, auditResearchLandscapeV3, resolveProcurementDocuments, renderResearchRecords } from '../scripts/skills/build-research-landscape-v3.mjs';
import { validateEpisode } from '../dist/skills/episode.js';
import { scoreResearchObjective } from '../src/skills/research-objective.ts';

const fullAnswer=expected=>({label:expected.label,unresolved:expected.unresolved,
 citations:expected.requiredEvidence.map(e=>({sourceId:e.sourceId,evidence:e.text}))});

test('small v3 packet has disjoint template support and query scenarios',()=>{
 const [episode]=buildResearchLandscapeV3();
 assert.deepEqual(validateEpisode(episode),[]);
 assert.equal(episode.support.cases.length,2);
 assert.equal(episode.query.cases.length,2);
 assert.equal(new Set(episode.source_groups).size,4);
 assert.ok(episode.support.cases.every(s=>episode.query.cases.every(q=>q.group!==s.group)));
 assert.deepEqual(episode.support.cases.map(c=>c.expected.label),['eligible','eligible']);
 assert.deepEqual(episode.query.cases.map(c=>c.expected.label),['ineligible','eligible']);
});

test('resolver reads only record-owned facts and every required source has a decision-changing ablation',()=>{
 const [episode]=buildResearchLandscapeV3();
 const audit=auditResearchLandscapeV3();
 assert.equal(audit.length,4);
  for(const row of [...episode.support.cases,...episode.query.cases]){
  const report=audit.find(x=>x.group===row.group);
  const docs=row.expected.documents;
  const recordDocs=row.services.research;
  assert.equal(resolveProcurementDocuments(JSON.parse(recordDocs.match(/PRIVATE_DOCUMENTS = (.*);\nexport/s)[1])),row.expected.label);
  assert.ok(row.expected.requiredEvidence.every(e=>docs.find(d=>d.id===e.sourceId)?.text.includes(e.text)));
  for(const sourceId of row.expected.requiredEvidence.map(e=>e.sourceId)){
  assert.ok(report.mutations.some(m=>m.sourceId===sourceId&&m.changesResult),`${row.id}: ${sourceId} lacks a pivotal raw-field mutation`);
  }
  assert.ok(report.mutations.every(m=>m.visibleTextChanged===true),`${row.id}: resolver pivot is not visible in rendered source prose`);
  assert.equal(scoreResearchObjective(JSON.parse(row.args[0]),fullAnswer(row.expected),row.expected).quality,1);
  const packet=JSON.parse(row.args[0]);
  assert.ok(!('expected' in packet)&&!('label' in packet)&&!('answer' in packet));
  assert.ok(docs.every(d=>!row.args.join('\n').includes(d.text)));
  const declaration=ts.transpileDeclaration(row.services.research,{fileName:'research.ts',compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
  assert.match(declaration,/search\(query: string\)/);assert.match(declaration,/read\(sourceId: string\)/);
  assert.ok(docs.every(d=>!declaration.includes(d.text)));
 }
});

test('policy prose tracks actual numeric fields and rendered mutations change the visible record',()=>{
 const [episode]=buildResearchLandscapeV3();
 for(const row of [...episode.support.cases,...episode.query.cases]){
  const raw=JSON.parse(row.services.research.match(/PRIVATE_DOCUMENTS = (.*);\nexport/s)[1]);
  const rule=raw.find(d=>d.type==='rule');
  assert.match(rule.text,/only bids from suppliers with active registration|every route requires a supplier whose registration is active/);
  assert.ok(rule.text.includes(`${rule.fields.standardCap} credits`));
  assert.ok(rule.text.includes(`${rule.fields.safetyCap} credits`));
  assert.ok(rule.text.includes(`${rule.fields.warehouseCap} credits`));
  assert.match(rule.text,/Northmere/);assert.match(rule.text,/signed waiver/);assert.match(rule.text,/depot is active/);
  const changed=structuredClone(raw),bid=changed.find(d=>d.type==='bid');
  bid.fields.amount+=1;
  const rendered=renderResearchRecords(changed).find(d=>d.type==='bid');
  assert.ok(rendered.text.includes(`${bid.fields.amount} credits`));
  assert.notEqual(rendered.text,bid.text);
 }
});

test('bad joins cannot be inferred into a model answer or silently treated as ineligible',()=>{
 const [episode]=buildResearchLandscapeV3(),sample=episode.support.cases[0];
 const docs=JSON.parse(sample.services.research.match(/PRIVATE_DOCUMENTS = (.*);\nexport/s)[1]);
 const join=docs.find(d=>d.type==='join'), before=join.text;
 join.fields.supplierFound=false;
 const rendered=renderResearchRecords(docs).find(d=>d.type==='join');
 assert.notEqual(rendered.text,before);
 assert.match(rendered.text,/supplier record vendor-B-104 found=false/);
 assert.equal(resolveProcurementDocuments(docs),null);
});
