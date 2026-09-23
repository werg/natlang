import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NativeNatlangHost } from '../../ts-host/dist/index.js';

const workspace = dirname(fileURLToPath(import.meta.url));
const host = new NativeNatlangHost({ workspace, network: true });
try {
  if (process.argv.includes('--install')) await host.environment.packages.prepareDependencies();
  const crisp = await host.run({ source: { kind: 'file', path: join(workspace, 'main.ts') }, inputs: {value:'42'} });
  let turn = 0;
  const natural = await host.run({ source: {kind:'file',path:join(workspace,'main.nl')},inputs:{value:'42'},
    options:{model:{max_turns:3}},modelTurn:()=> ++turn === 1
      ? {calls:[['eval',{code:'import isNumber from "is-number"; return isNumber(value);'}]]}
      : {calls:[['mark_lines',{start:1}]]} });
  if (crisp.outcome.kind !== 'done' || crisp.value !== true || natural.outcome.kind !== 'done' || natural.value !== true)
    throw new Error(JSON.stringify({crisp,natural}));
  console.log(JSON.stringify({handwritten:crisp.value,natural:natural.value,workspace}));
} finally { host.close(); }
