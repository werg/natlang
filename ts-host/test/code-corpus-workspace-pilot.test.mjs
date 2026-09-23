import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

test('workspace pilot captures a real unit-test call and replays an imported dependency', async t => {
  const root = await mkdtemp(join(tmpdir(),'corpus-workspace-pilot-'));
  t.after(() => rm(root,{recursive:true,force:true}));
  const workspace = join(root,'workspace'), output = join(root,'pilot');
  await mkdir(join(workspace,'src'),{recursive:true});
  await mkdir(join(workspace,'test'),{recursive:true});
  await mkdir(join(workspace,'node_modules','fixture-helper'),{recursive:true});
  await writeFile(join(workspace,'package.json'),JSON.stringify({name:'fixture-workspace',private:true,type:'module',dependencies:{'fixture-helper':'1.0.0'}}));
  await writeFile(join(workspace,'node_modules','fixture-helper','package.json'),JSON.stringify({name:'fixture-helper',version:'1.0.0',type:'module',exports:'./index.js'}));
  await writeFile(join(workspace,'node_modules','fixture-helper','index.js'),'export const twice = value => value * 2; export const triple = value => value * 3;\n');
  await writeFile(join(workspace,'src','math.mjs'),"import { twice, triple } from 'fixture-helper';\n/** Double an integer. */\nexport function double(value) { return twice(value); }\n/** Triple an integer. */\nexport function tripleValue(value) { return triple(value); }\n");
  await writeFile(join(workspace,'test','math.test.mjs'),"import test from 'node:test';\nimport assert from 'node:assert/strict';\nimport { double, tripleValue } from '../src/math.mjs';\ntest('arithmetic helpers', () => { assert.equal(double(4), 8); assert.equal(double(7), 14); assert.equal(tripleValue(3), 9); });\n");
  await writeFile(join(workspace,'package-lock.json'),JSON.stringify({name:'fixture-workspace',lockfileVersion:3,packages:{'':{dependencies:{'fixture-helper':'1.0.0'},devDependencies:{}},'node_modules/fixture-helper':{version:'1.0.0'}}}));
  const result = spawnSync(process.execPath,['scripts/code-corpus/workspace-pilot.mjs','--execute','--workspace',workspace,
    '--source','src/math.mjs','--test','test/math.test.mjs','--function','double','--function','tripleValue','--output',output],
  {cwd:new URL('..',import.meta.url),encoding:'utf8',timeout:60000});
  assert.equal(result.status,0,result.stderr || result.stdout);
  const manifest = JSON.parse(await readFile(join(output,'manifest.json'),'utf8'));
  assert.equal(manifest.execution.selected_function_captures,3,JSON.stringify({manifest,captures:await readFile(join(output,'captures.jsonl'),'utf8')}));
  assert.equal(manifest.native_replay.accepted,3, result.stdout);
  assert.equal(manifest.native_replay.turns,6, result.stdout);
  assert.ok(manifest.native_replay.accepted>0);
  assert.equal(manifest.package.name,'fixture-workspace');
  assert.deepEqual(manifest.functions,['double','tripleValue']);
  assert.ok(Object.values(manifest.imports_by_function).every(imports=>imports.every(item=>item.specifier==='fixture-helper')));
  assert.ok(manifest.package.lockfile_sha256['package-lock.json']);
  const captureRows = (await readFile(join(output,'captures.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(captureRows.map(row=>row.args),[[4],[7],[3]]);
  assert.ok(captureRows.every(row=>row.portable && row.outcome==='return'));
});
