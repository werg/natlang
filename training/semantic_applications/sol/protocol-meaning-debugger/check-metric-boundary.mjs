/** Exact qualification controls for authored metric handoff; no NL execution credit. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {verifyCandidate} from './verify-result.mjs';
const manifest=JSON.parse(await readFile(new URL('./project.json',import.meta.url),'utf8'));
const read=async path=>JSON.parse(await readFile(path,'utf8'));
const value=await read(manifest.reference_path),oracle=await read(manifest.oracle_path);
const inputs={request:await read(manifest.request_path),input:await read(manifest.snapshot_path)};
const fixture=Object.fromEntries(await Promise.all(manifest.fixture_files.map(async path=>[path,await readFile(manifest.fixture_root+'/'+path,'utf8')])));
const run=candidate=>verifyCandidate({value:candidate,oracle,inputs,fixture});
assert.equal(run(value).ok,true);
const reordered=structuredClone(value);reordered.baseline.right.metric.localDecisions.reverse();
assert.equal(run(reordered).ok,true);
for(const [label,change] of [
 ['missing symbolic reading',v=>delete v.baseline.right.metric.reading],
 ['child owner mismatch',v=>v.baseline.right.metric.reading.implementationId='L'],
 ['erased display mismatch',v=>v.baseline.right.metric.displayLabelCompatibility='compatible'],
 ['absent counted proposition',v=>v.baseline.right.metric.countedMeaning=null],
 ['missing parent contribution',v=>v.baseline.right.metric.localDecisions=v.baseline.right.metric.localDecisions.filter(d=>d.site!=='metrics.nl')],
 ['unsupported reading quotation',v=>v.baseline.right.metric.reading.witnesses[0].quote='Absent quotation.'],
 ['repaired display mismatch',v=>v.repair.baseline.right.metric.displayLabelCompatibility='misleading'],
 ['binding changes executable polarity',v=>v.baseline.right.executable.numeratorOutcome='pass'],
]){const candidate=structuredClone(value);change(candidate);assert.equal(run(candidate).ok,false,label);}
console.log(JSON.stringify({positive:2,negative:8,ok:true,scope:'Authored typed metric witness controls; semantic entailment and actual stage execution remain pending.'}));
