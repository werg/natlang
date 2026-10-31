import { externalModule } from '../native/external.js';
import { createNatlangRuntime } from '../runtime/runtime.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { nodePlatform } from '../runtime/node.js';
import { bindServiceAliases } from './service-aliases.js';
import { checkReportCompleteness } from './report-completeness.js';
import { executeProgram } from './collector.js';

/** Executable probes for collector semantics that must match the source rows being run. */
export async function teacherCollectorCapabilities(): Promise<Record<string, unknown>> {
  const source = externalModule('source', `export function read(): number { return 42; }
    export function write(): number { return 99; }`);
  const modules = new Map([['source', source]]);
  const services: Record<string, object> = { source: source.exports };
  const declarations: Record<string, string> = { source: source.declaration };
  bindServiceAliases(modules, { restricted: { source: 'source', methods: ['read'] } }, services, declarations);
  const restricted = services.restricted as { read(): number; write?: () => number };
  if (restricted.read() !== 42 || Object.hasOwn(services, 'source') ||
      Object.hasOwn(restricted, 'write') || !declarations.restricted?.includes('function read(): number') ||
      declarations.restricted.includes('function write()'))
    throw new Error('service alias capability probe failed');

  const receipt = { receiptId: 'receipt-1', sequence: 1, toolName: 'refuelData', requestor: 'assistant',
    arguments: { gbAmount: 2 }, value: { charge: 4 } };
  const recorded = { actionId: 'action-1', receiptId: receipt.receiptId, sequence: receipt.sequence,
    toolName: receipt.toolName, requestor: receipt.requestor, arguments: receipt.arguments, result: receipt.value };
  const spec = { schema: 'natlang.report-completeness/2' as const, mutating_methods: ['apply'],
    effect_receipt_path: 'receipt',
    report_effects_path: 'report.effects', checkpoint_effects_path: 'report.checkpoint.receipts',
    report_diagnosis_path: 'report.diagnosis', diagnosis_function: 'presentDiagnosis.nl' };
  const wrapped = { applied: true, receipt: { receiptId: receipt.receiptId, actionId: 'action-1',
    sequence: receipt.sequence, toolName: receipt.toolName, requestor: receipt.requestor,
    arguments: receipt.arguments, result: receipt.value }, blockers: [] };
  const value = { report: { effects: [recorded], checkpoint: { receipts: [recorded] }, diagnosis: { cause: 'capacity' } } };
  const graph = [{ kind: 'effect', phase: 'completed', method: 'apply', result: wrapped }];
  const ledger = [{ host_result: { capture_kind: 'invocation_output', complete: true,
    definition_source: 'presentDiagnosis.nl', value: { cause: 'capacity' } } }];
  const positive = checkReportCompleteness(spec, value, graph, ledger);
  const negative = checkReportCompleteness(spec, { report: { ...value.report, effects: [] } }, graph, ledger);
  if (!positive.required || !positive.accepted || negative.required !== true || negative.accepted !== false)
    throw new Error('report completeness capability probe failed');

  // Exercise the production runtime scope filter with a file-backed child: it gets its namespace,
  // while its caller and an unrelated helper do not.
  const root = loadVirtualNatlang({
    'root.nl': '---\nargs: {}\nreturns: boolean\n---\nCall the scoped child.',
    'root/deviceRead.nl': '---\nargs: {}\nreturns: boolean\n---\nRead the scoped device state.',
    'root/accountRead.nl': '---\nargs: {}\nreturns: boolean\n---\nRead account state.',
  }, 'root.nl');
  const visible: Record<string, string[]> = {};
  const runtime = createNatlangRuntime({ platform: nodePlatform(), services: { device: { read: () => 7 }, telecom: { read: () => 9 } },
    serviceDeclarations: { device: 'export function read(): number;', telecom: 'export function read(): number;' },
    serviceScopes: { device: ['root/deviceRead.nl'], telecom: ['root/accountRead.nl'] },
    agent: async session => {
      const name = session.lam.functionName;
      visible[name] = Object.keys(session.availableServices()).sort();
      if (name === 'root') {
        const result = await session.applyAsync('eval', { code: 'await deviceRead(); return true;', finish: true });
        if (result.kind === 'rejected') throw new Error(result.text);
      } else {
        const result = await session.applyAsync('eval', { code: 'const n = await device.read(); return n === 7;', finish: true });
        if (result.kind === 'rejected') throw new Error(result.text);
      }
    } });
  const rootResult = await runtime.run(() => root());
  if (rootResult !== true || JSON.stringify(visible.root) !== '[]' ||
      JSON.stringify(visible.deviceRead) !== '["device"]')
    throw new Error('service scope capability probe failed');

  // Exercise the actual production collector route, including controller binding,
  // task-frame preservation, model-driven Natlang child execution and graph export.
  const controllerRun = await executeProgram({ version: 'natlang.program/2', id: 'capability-probe', kind: 'lambda_source',
    semantics: { root: 'entry.nl', files: {
      'natlang.json': '{}',
      'entry.nl': '---\nargs:\n  value: number\nreturns: number\nuses: [workflow]\n---\nRun the typed controller.\n',
      'entry/advance.nl': '---\nargs:\n  value: number\nreturns: number\n---\nAdvance the workflow.\n',
      'entry/advance/step.nl': '---\nargs:\n  value: number\nreturns: number\n---\nReturn value plus one.\n',
      'workflow.ts': 'export async function run(value: number, operations: { step(value: number): Promise<number> }): Promise<number> { return await operations.step(value); }\n',
    }, inputs: { value: 41 }, expected: 42, oracle: 'exact',
      application_controller: { schema: 'natlang.application-controller/1', module: 'workflow', export: 'run',
        operations: 'advance', input_arguments: ['value'] } } },
  async () => ({ calls: [['eval', { code: 'return value + 1;', finish: true }]] }),
  { rootSeed: 5, systemPrompt: '', contextTokens: 4096, maxTurns: 3, runId: 'collector-controller-capability-probe' });
  const controllerGraph = controllerRun.outcome.execution_graph as
    { definition?: { id?: string } }[] | undefined;
  if (controllerRun.outcome.accepted !== true || controllerRun.outcome.value !== 42 ||
      !controllerGraph?.some(event => event.definition?.id === 'nl:entry/advance/step.nl'))
    throw new Error('application controller collector probe did not execute and export its declared Natlang operation');

  return { schema: 'natlang.teacher-collector-capabilities/1',
    service_aliases: 'natlang.service-aliases/2', service_scopes: 'natlang.service-scopes/1',
    report_completeness: 'natlang.report-completeness/2', application_controller: 'natlang.application-controller/1',
    probes: { service_aliases: true, service_scopes: true, report_completeness: true, application_controller: true } };
}
