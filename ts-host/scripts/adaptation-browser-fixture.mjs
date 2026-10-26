import { writeFileSync } from 'node:fs';
import * as host from '../dist/index.js';
import { candidateArtifact } from '../dist/evaluation/runner.js';
const files = { 'main.ts': "import { nl } from '@natlang/node';\nexport const judge = /* @natlangSite portable */ nl<number>`Return one.`;\nexport async function invoke(): Promise<number> { await Promise.resolve(); return judge(); }\n" };
// The browser smoke recompiles these same sources with its browser compiler.
// Adaptation identity includes the compiler's program identity, so generate
// this artifact for that exact target and let bindAdaptation enforce it.
const project = host.compileVirtualProject({ files }, host, { target: 'browser' });
if (!project.ok) throw new Error(host.formatDiagnostics(project.diagnostics));
const program = project.manifest.adaptation;
const executor = { id: 'portable-fixture', configuration: { revision: 1 } };
const component = program.components.find(component => component.origin === 'authored-inline');
const artifact = candidateArtifact({ program, components: [component.key], suite: { executorIdentity: executor }, suiteHash: host.fingerprint('portable-fixture') },
  { [component.key]: { kind: 'lambda.instructions', template: { segments: ['Return nine.'], slotIds: [] } } });
// Keep the generator's own executable sanity check on the Node target. The
// serialized artifact above intentionally belongs to the browser target.
// Give the Node-target program its own identity as a separately compiled
// target build; it must never bind to the browser project's descriptor.
const nodeProject = host.compileVirtualProject({ files }, host, { target: 'node', programId: 'node-target' });
if (!nodeProject.ok) throw new Error(host.formatDiagnostics(nodeProject.diagnostics));
const nodeProgram = nodeProject.manifest.adaptation;
const nodeComponent = nodeProgram.components.find(component => component.origin === 'authored-inline');
const nodeArtifact = candidateArtifact({ program: nodeProgram, components: [nodeComponent.key], suite: { executorIdentity: executor }, suiteHash: host.fingerprint('portable-fixture') },
  { [nodeComponent.key]: { kind: 'lambda.instructions', template: { segments: ['Return nine.'], slotIds: [] } } });
const binding = host.bindAdaptation(nodeArtifact, nodeProgram, executor);
const runtime = host.createNatlangRuntime({ program: nodeProgram, adaptation: binding, executorIdentity: executor,
  agent: session => { session.lam.return = session.lam.body.includes('nine') ? 9 : 1; } });
const app = nodeProject.require('main.ts');
const values = await Promise.all([runtime.run(() => app.invoke()), runtime.run(() => app.invoke(), { adaptation: null })]);
if (JSON.stringify(values) !== '[9,1]') throw new Error('Node artifact fixture failed');
writeFileSync(new URL('../test/fixtures/adaptation-browser.json', import.meta.url),
  JSON.stringify({ files, program, artifact, nodeProgram, nodeArtifact, executor }, null, 2) + '\n');
console.log('PASS Node portable artifact fixture');
