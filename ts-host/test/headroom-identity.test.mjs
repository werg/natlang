import test from 'node:test';
import assert from 'node:assert/strict';
import {headroomIdentity, resumedHeadroomRows} from '../scripts/skills/headroom-identity.mjs';
const context = {executorId:'executor', pins:{host:'hash'}, inputSha256:'input', options:{band:'0,0.95', split:'train', limit:'0', concurrency:'4', 'family-probe':'0'}};
test('resume identity binds every selection and execution allocation option', () => {
  const identity = headroomIdentity(context);
  for (const [key,value] of Object.entries({band:'0.2,0.8',split:'all',limit:'2',concurrency:'2','family-probe':'6','database-root':'other-db','arena-root':'other-arena'}))
    assert.notEqual(headroomIdentity({...context, options:{...context.options,[key]:value}}), identity, key);
  assert.equal(headroomIdentity({...context,options:{...context.options,band:'0.0,0.950'}}),identity);
});
test('resume rejects mixed, duplicated, truncated and foreign identities before calls', () => {
  const identity = headroomIdentity(context), row={schema:'natlang.episode-headroom/2',screen:identity,episode:'a'};
  assert.equal(resumedHeadroomRows(JSON.stringify(row)+'\n', identity).size,1);
  assert.throws(()=>resumedHeadroomRows(JSON.stringify({...row,screen:'old'}),identity),/different/);
  assert.throws(()=>resumedHeadroomRows([row,row].map(JSON.stringify).join('\n'),identity),/Duplicate/);
  assert.throws(()=>resumedHeadroomRows('{',identity));
});
test('invalid allocations and bands fail closed', () => {
  for (const [key,value] of [['concurrency','0'],['concurrency','NaN'],['limit','-1'],['family-probe','1.5'],['band','0,0.9,1'],['split','anything']])
    assert.throws(()=>headroomIdentity({...context,options:{...context.options,[key]:value}}));
});
