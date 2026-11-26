import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { modelConnectionAttemptTimeout } from '../dist/model/node-connection-policy.js';
test('address attempt window is configurable without changing request deadlines', () => {
 assert.equal(modelConnectionAttemptTimeout({}),1500);
 assert.equal(modelConnectionAttemptTimeout({NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS:'0'}),undefined);
 for(const value of ['-1','1','NaN','1.5','60001']) assert.throws(()=>modelConnectionAttemptTimeout({NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS:value}));
});
for (const [initial,override,expected] of [[250,'',1500],[600,'',600],[250,'0',250],[600,'2200',2200]])
 test(`native process policy: ${initial}/${override} -> ${expected}`, () => {
  const module = new URL('../dist/model/node-connection-runtime.js',import.meta.url).href;
  const program = `import * as net from 'node:net'; net.setDefaultAutoSelectFamilyAttemptTimeout(${initial}); const {configureNodeModelConnections}=await import(${JSON.stringify(module)}); await configureNodeModelConnections(); console.log(net.getDefaultAutoSelectFamilyAttemptTimeout());`;
  const result=execFileSync(process.execPath,['--input-type=module','-e',program],{encoding:'utf8',env:{...process.env,NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS:override}});
  assert.equal(Number(result.trim()),expected);
 });

for(const flag of ['--version','--help']) test('offline CLI '+flag+' remains usable with invalid network configuration', () => {
 const cli=new URL('../dist/cli/main.js',import.meta.url);
 const text=execFileSync(process.execPath,[cli.pathname,flag],{encoding:'utf8',env:{...process.env,NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS:'invalid'}});
 assert.ok(text.trim());
});
