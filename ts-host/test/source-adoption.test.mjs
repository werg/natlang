import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,readFileSync,existsSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {Folder,adoptSource,rollbackSource} from '../dist/index.js';
const manifest=(before,after)=>({schema:'natlang.program-source/1',base:Folder.fromFiles(before).snapshot().digest,source:Folder.fromFiles(after).snapshot().digest,baseFiles:before,files:after,contract:{}});
const verify=async source=>({source:source.digest,valid:true,diagnostics:[],contract:{entry:'main.ts',exportName:'solve',programId:'adopt'}});

test('source adoption handles create/delete, rejects stale contents, and rolls back exact source',async()=>{
 const directory=mkdtempSync(join(tmpdir(),'natlang-adopt-'));
 const before={'main.ts':'before','old.ts':'remove'},after={'main.ts':'after','helper.ts':'new'};
 for(const [path,text]of Object.entries(before))writeFileSync(join(directory,path),text);
 writeFileSync(join(directory,'unrelated.txt'),'preserve');
 const record=await adoptSource(directory,manifest(before,after),verify,join(directory,'adoption.json'));
 assert.equal(readFileSync(join(directory,'main.ts'),'utf8'),'after');assert.equal(existsSync(join(directory,'old.ts')),false);
 assert.equal(readFileSync(join(directory,'helper.ts'),'utf8'),'new');assert.equal(readFileSync(join(directory,'unrelated.txt'),'utf8'),'preserve');
 await assert.rejects(()=>adoptSource(directory,manifest(before,after),verify),/stale checkout/);
 rollbackSource(record);assert.equal(readFileSync(join(directory,'main.ts'),'utf8'),'before');assert.equal(existsSync(join(directory,'helper.ts')),false);
});

test('failed verification and mutations during verification publish no source',async()=>{
 const directory=mkdtempSync(join(tmpdir(),'natlang-adopt-'));writeFileSync(join(directory,'main.ts'),'before');
 const source=manifest({'main.ts':'before'},{'main.ts':'after'});
 await assert.rejects(()=>adoptSource(directory,source,async value=>({...await verify(value),valid:false,diagnostics:['type error']})),/build failed/);
 assert.equal(readFileSync(join(directory,'main.ts'),'utf8'),'before');
 await assert.rejects(()=>adoptSource(directory,source,async value=>{writeFileSync(join(directory,'main.ts'),'concurrent');return verify(value);}),/stale checkout during/);
 assert.equal(readFileSync(join(directory,'main.ts'),'utf8'),'concurrent');
});

test('recorded interrupted publication can finish without overwriting unrelated edits',async()=>{
 const {recoverAdoption}=await import('../dist/index.js');
 const directory=mkdtempSync(join(tmpdir(),'natlang-recover-')),before={'main.ts':'before','helper.ts':'old'},after={'main.ts':'after','helper.ts':'new'};
 const record={schema:'natlang.source-adoption/1',directory,before,after,beforeDigest:manifest(before,after).base,afterDigest:manifest(before,after).source,changed:Object.keys(before),status:'prepared'};
 const path=join(directory,'adoption.json');writeFileSync(path,JSON.stringify(record));
 writeFileSync(join(directory,'main.ts'),'after');writeFileSync(join(directory,'helper.ts'),'old');
 recoverAdoption(path);assert.equal(readFileSync(join(directory,'helper.ts'),'utf8'),'new');
 writeFileSync(join(directory,'helper.ts'),'user edit');assert.throws(()=>recoverAdoption(path),/stale checkout/);
});
