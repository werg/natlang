/** Fresh realm per target case. The evaluator capability is never installed here. */
import { parentPort, workerData } from 'node:worker_threads';
import * as runtime from '../runtime/node.js';
import * as targetRuntime from './target-runtime.js';
import {TOOLS_PROMPT} from '../native/prompt.js';
import { Folder,APPLY_TO_FOLDER } from '../native/scoped-fs.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { compileVirtualProject } from '../runtime/virtual-project.js';
import type { ModelTurn } from '../contracts.js';
class FixtureError extends Error {}
const pending = new Map<number, { resolve: (turn: ModelTurn) => void; reject: (error: Error) => void }>();
let sequence = 0;
parentPort!.on('message', message => {
  const request = pending.get(message.id);
  if (request) { pending.delete(message.id); message.error ? request.reject(new Error(message.error)) : request.resolve(message.turn); }
});
try {
  const build = compileVirtualProject({ files: workerData.files }, targetRuntime, { programId: workerData.contract.programId, constrained: true, target: 'node' });
  if (!build.ok) throw new Error('target build failed');
  const target = workerData.contract.entry.endsWith('.nl') ? loadVirtualNatlang(workerData.files, workerData.contract.entry) : build.require(workerData.contract.entry)[workerData.contract.exportName];
  const services = Object.fromEntries(Object.entries(workerData.services ?? {}).map(([name, source]) => {
    const serviceBuild = compileVirtualProject({ files: { 'service.ts': source as string } }, runtime, { constrained: true, target: 'node' });
    if (!serviceBuild.ok) throw new FixtureError('fixture service failed compilation: '+name+'\n'+serviceBuild.diagnostics.filter(row=>row.severity==='error').map(row=>row.message).join('\n'));
    return [name, serviceBuild.require('service.ts')];
  }));
  if (typeof target !== 'function') throw new Error('target entry is not callable');
  const task = runtime.createNatlangRuntime({ model: {driver:request => new Promise((resolve, reject) => {
    const id = sequence++; pending.set(id, { resolve, reject }); parentPort!.postMessage({ type: 'request', id, request });
  })}, trace:trace=>parentPort!.postMessage({type:'trace',callId:trace.callId,events:trace.events.filter(event=>event.kind==='action'||event.kind==='model_request')}), services, serviceDeclarations: workerData.services, network: false, codeEdits: 'deny', seed: { mode: 'derived', root: workerData.seed }, limits: workerData.limits });
  const folder=workerData.folder?Folder.fromFiles(workerData.folder):undefined;
  const value = await task.run(() => folder ? (typeof (target as unknown as Record<PropertyKey,unknown>)[APPLY_TO_FOLDER]==='function'?folder.apply(target, ...workerData.args):target(folder,...workerData.args)) : target(...workerData.args));
  parentPort!.postMessage({ type: 'result', value: JSON.parse(JSON.stringify(value)),...(folder?{files:Object.fromEntries(folder.filePaths().map(path=>[path,new TextDecoder().decode(folder.readBytesSync(path))]))}:{}) });
} catch (error) {
  parentPort!.postMessage({ type: 'result', failureKind:error instanceof FixtureError?'fixture':'target', error: error instanceof Error ? error.message : String(error) });
}
parentPort!.close();
