import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {gzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {jsonlLines,jsonlRows} from '../scripts/jsonl-stream.mjs';

test('JSONL preserves Unicode separators, CRLF, UTF8 chunk boundaries and final unterminated rows',async()=>{
 const dir=mkdtempSync(join(tmpdir(),'natlang-jsonl-'));
 try{
  const rows=[{text:'x'.repeat(65530)+'💡\u2028inside\u2029text'}, {text:'final'}];
  const content=JSON.stringify(rows[0])+'\r\n\n'+JSON.stringify(rows[1]);
  for(const gz of [false,true]){
   const path=join(dir,gz?'rows.jsonl.gz':'rows.jsonl');writeFileSync(path,gz?gzipSync(content):content);
   assert.deepEqual(await Array.fromAsync(jsonlRows(path)),rows);
   assert.deepEqual(await Array.fromAsync(jsonlLines(path)),[JSON.stringify(rows[0]),'',JSON.stringify(rows[1])]);
  }
 }finally{rmSync(dir,{recursive:true,force:true})}
});

test('trajectory converter keeps complete Unicode-containing rows in both passes',()=>{
 const dir=mkdtempSync(join(tmpdir(),'natlang-convert-unicode-'));
 try{
  const row={id:'unicode:decision:0000',source_ref:{trajectory_id:'unicode'},split:'train',messages:[{role:'user',content:'Read A\u2028B\u2029C.'}],target:{role:'assistant',content:'A\u2028B\u2029C'}};
  const input=join(dir,'input.jsonl'),output=join(dir,'output.jsonl'),summary=join(dir,'summary.json');
  writeFileSync(input,JSON.stringify(row)+'\n');
  execFileSync(process.execPath,[resolve(import.meta.dirname,'../scripts/neuralese-convert-trajectories.mjs'),'--out',output,'--pieces',join(dir,'pieces.jsonl'),'--summary',summary,input],{stdio:'pipe'});
  const stats=JSON.parse(readFileSync(summary,'utf8'));
  assert.equal(stats.records,1);assert.equal(stats.unreadable,0);
  const result=JSON.parse(readFileSync(output,'utf8'));
  assert.equal(result.id,row.id);assert.equal(result.target.content,row.target.content);
 }finally{rmSync(dir,{recursive:true,force:true})}
});

// The rewriting stage shares this framing contract and preserves target bytes.
test('trajectory rewrite preserves Unicode-bearing native rows',()=>{
 const dir=mkdtempSync(join(tmpdir(),'natlang-rewrite-unicode-'));
 try{
  const row={id:'unicode:decision:0000',messages:[{role:'user',content:'A\u2028B\u2029C'}],target:{role:'assistant',content:'A\u2028B\u2029C'}};
  const input=join(dir,'input.jsonl'),output=join(dir,'output.jsonl'),summary=join(dir,'summary.json');
  writeFileSync(input,JSON.stringify(row)+'\n');
  execFileSync(process.execPath,[resolve(import.meta.dirname,'../scripts/neuralese-rewrite-trajectories.mjs'),'--out',output,'--summary',summary,input],{stdio:'pipe'});
  const stats=JSON.parse(readFileSync(summary,'utf8'));
  assert.equal(stats.records,1);assert.equal(stats.unreadable,0);
  const result=JSON.parse(readFileSync(output,'utf8'));
  assert.deepEqual(result.messages,row.messages);assert.deepEqual(result.target,row.target);
 }finally{rmSync(dir,{recursive:true,force:true})}
});
