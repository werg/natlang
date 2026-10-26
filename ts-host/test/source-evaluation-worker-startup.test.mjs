import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const run = promisify(execFile);

test('source evaluation inherits safe Node options under V8 flags and an input-type parent', async () => {
  const modules = {
    folder: new URL('../dist/native/scoped-fs.js', import.meta.url).href,
    host: new URL('../dist/improvement/host.js', import.meta.url).href,
    usage: new URL('../dist/evaluation/usage.js', import.meta.url).href,
  };
  const code = [
    'import { Folder } from ' + JSON.stringify(modules.folder) + ';',
    'import { SourceEvaluator } from ' + JSON.stringify(modules.host) + ';',
    'import { UsageGateway } from ' + JSON.stringify(modules.usage) + ';',
    'const source = Folder.fromFiles({"solve.ts":"export function solve(): number { return 7; }"}).snapshot();',
    'const evaluator = new SourceEvaluator({entry:"solve.ts",exportName:"solve",programId:"worker-startup"},',
    '[{id:"one",group:"one",split:"train",args:[],expected:7}],',
    '()=>{throw new Error("crisp target must not request a model");},',
    'new UsageGateway({maxModelCalls:1,maxRollouts:1,maxProposals:0}),{executorId:"worker-startup"});',
    'const report = await evaluator.evaluate(source,{split:"train"});',
    'if (report.quality !== 1 || report.outcomes[0].value !== 7) throw new Error(JSON.stringify(report.outcomes));',
    'process.stdout.write("worker-result:7");',
  ].join('\n');
  const {stdout} = await run(process.execPath, ['--stack-trace-limit=25', '--input-type=module', '--eval', code],
    {env:process.env,maxBuffer:1024*1024});
  assert.match(stdout,/worker-result:7/);
});
