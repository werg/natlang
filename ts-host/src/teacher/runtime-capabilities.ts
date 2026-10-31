import { externalModule } from '../native/external.js';
import { createNatlangRuntime } from '../runtime/runtime.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { nodePlatform } from '../runtime/node.js';
import { bindServiceAliases } from './service-aliases.js';
import { checkReportCompleteness } from './report-completeness.js';

/** Executable probes for collector semantics that must match the source rows being run. */
export async function teacherCollectorCapabilities(): Promise<Record<string, unknown>> {
  const source = externalModule('source', 'export function read(): number { return 42; }');
  const modules = new Map([['source', source]]);
  const services: Record<string, object> = { source: source.exports };
  const declarations: Record<string, string> = { source: source.declaration };
  bindServiceAliases(modules, { restricted: { source: 'source', methods: ['read'] } }, services, declarations);
  if ((services.restricted as { read(): number }).read() !== 42 || Object.hasOwn(services, 'source') ||
      !declarations.restricted?.includes('function read(): number'))
    throw new Error('service alias capability probe failed');

  const receipt = { receiptId: 'receipt-1', sequence: 1, toolName: 'refuelData', requestor: 'assistant',
    arguments: { gbAmount: 2 }, value: { charge: 4 } };
  const recorded = { actionId: 'action-1', receiptId: receipt.receiptId, sequence: receipt.sequence,
    toolName: receipt.toolName, requestor: receipt.requestor, arguments: receipt.arguments, result: receipt.value };
  const spec = { schema: 'natlang.report-completeness/1' as const, mutating_methods: ['refuelData'],
    report_effects_path: 'report.effects', checkpoint_effects_path: 'report.checkpoint.receipts',
    report_diagnosis_path: 'report.diagnosis', diagnosis_function: 'presentDiagnosis.nl' };
  const value = { report: { effects: [recorded], checkpoint: { receipts: [recorded] }, diagnosis: { cause: 'capacity' } } };
  const graph = [{ kind: 'effect', phase: 'completed', method: 'refuelData', result: receipt }];
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

  return { schema: 'natlang.teacher-collector-capabilities/1',
    service_aliases: 'natlang.service-aliases/1', service_scopes: 'natlang.service-scopes/1',
    report_completeness: 'natlang.report-completeness/1',
    probes: { service_aliases: true, service_scopes: true, report_completeness: true } };
}
