import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = resolve(process.argv[2]);
const output = resolve(process.argv[3]);
const metadata = JSON.parse(await readFile(root + '/metadata.json', 'utf8'));
const binding = metadata.external_services.telecom;
const fixtures = (await readFile(root + '/' + binding.fixtures_path, 'utf8'))
  .split('\n').filter(Boolean).map(line => JSON.parse(line));
const fixtureRow = fixtures.find(row => row.id === 'tau2-telecom-train-070-v12-payment-no-consent');
assert.ok(fixtureRow, 'public TRAIN fixture exists');
let source = await readFile(root + '/' + binding.source_path, 'utf8');
assert.equal(source.split('/*__NATLANG_WORLD_FIXTURE__*/').length - 1, 1);
source = source.replace('/*__NATLANG_WORLD_FIXTURE__*/', JSON.stringify(fixtureRow.fixture))
  .replace('/*__NATLANG_CODEBASE_ROOT__*/', JSON.stringify(root));
for (const tree of metadata.collector_runtime.trees) {
  const name = tree.name.replace(/[^A-Za-z0-9_]/g, '_');
  const marker = '/*__NATLANG_RUNTIME_TREE_' + name + '__*/';
  assert.equal(source.split(marker).length - 1, 1, 'runtime tree marker ' + name);
  source = source.replace(marker, JSON.stringify(resolve(tree.root)));
}
const scratch = resolve(process.env.NATLANG_RECEIPT_PREFLIGHT_WORKROOT ?? dirname(output) + '/runtime');
await mkdir(scratch, { recursive: true });
await writeFile(scratch + '/package.json', '{"type":"module"}\n');
const boundPath = scratch + '/collector-service.ts';
await writeFile(boundPath, source);
const spool = resolve(process.env.NATLANG_TEACHER_JOURNAL_SPOOL_ROOT ?? root + '/diagnostics/.receipt-preflight-spool');
await mkdir(spool, { recursive: true });
process.env.NATLANG_TEACHER_JOURNAL_SPOOL_ROOT = spool;
const service = await import(pathToFileURL(boundPath).href);
const records = [];
const call = async (name, result) => {
  const receipt = await result;
  assert.equal(typeof receipt.receiptId, 'string');
  assert.equal(typeof receipt.sequence, 'number');
  assert.equal(typeof receipt.caseSessionId, 'string');
  assert.equal(typeof receipt.toolName, 'string');
  assert.ok(receipt.requestor === 'assistant' || receipt.requestor === 'user');
  assert.ok(receipt.arguments && typeof receipt.arguments === 'object' && !Array.isArray(receipt.arguments));
  records.push({ method: name, tool: receipt.toolName, requestor: receipt.requestor, argument_keys: Object.keys(receipt.arguments).sort(),
    sequence: receipt.sequence, value_type: typeof receipt.value });
  return receipt;
};
const expected = {
  lookupCustomerByPhone: ['phone_number'], getLineDetails: ['id'], getPlanDetails: ['id'],
  getBillDetails: ['id'], getDataUsage: ['customer_id', 'line_id'],
  checkWifiStatus: [], checkWifiCallingStatus: [], checkInstalledApps: [], checkAppStatus: ['app_name'],
  checkAppPermissions: ['app_name'], canSendMms: [], checkStatusBar: [], checkNetworkStatus: [],
  checkNetworkModePreference: [], checkSimStatus: [], checkDataRestrictionStatus: [], checkApnSettings: [],
  checkVpnStatus: [], checkPaymentRequest: [], runSpeedTest: [],
};
const deviceMethods = new Set([
  'checkWifiStatus', 'checkWifiCallingStatus', 'checkInstalledApps', 'checkAppStatus', 'checkAppPermissions',
  'canSendMms', 'checkStatusBar', 'checkNetworkStatus', 'checkNetworkModePreference', 'checkSimStatus',
  'checkDataRestrictionStatus', 'checkApnSettings', 'checkVpnStatus', 'checkPaymentRequest', 'runSpeedTest',
]);
const customer = (await call('lookupCustomerByPhone', service.lookupCustomerByPhone(fixtureRow.fixture.request.callerPhone))).value;
const lines = [];
for (const lineId of customer.lineIds) lines.push(await call('getLineDetails', service.getLineDetails(lineId)));
const match = lines.filter(receipt => receipt.arguments.id === receipt.value.lineId &&
  receipt.value.phoneNumber.replace(/\D/g, '') === fixtureRow.fixture.request.callerPhone.replace(/\D/g, ''));
assert.equal(match.length, 1, 'actual host receipts select exactly the caller-matching line');
for (const planId of [...new Set(lines.map(receipt => receipt.value.planId))])
  await call('getPlanDetails', service.getPlanDetails(planId));
for (const billId of customer.billIds) await call('getBillDetails', service.getBillDetails(billId));
for (const line of lines) await call('getDataUsage', service.getDataUsage(customer.customerId, line.value.lineId));
const deviceReads = [
  ['checkWifiStatus', []], ['checkWifiCallingStatus', []], ['checkInstalledApps', []],
  ['checkAppStatus', ['messaging']], ['checkAppPermissions', ['messaging']], ['canSendMms', []],
  ['checkStatusBar', []], ['checkNetworkStatus', []], ['checkNetworkModePreference', []], ['checkSimStatus', []],
  ['checkDataRestrictionStatus', []], ['checkApnSettings', []], ['checkVpnStatus', []],
  ['checkPaymentRequest', []], ['runSpeedTest', []],
];
for (const [name, args] of deviceReads) await call(name, service[name](...args));
for (const record of records) {
  assert.deepEqual(record.argument_keys, expected[record.method], record.method + ' actual argument keys');
  assert.equal(record.requestor, deviceMethods.has(record.method) ? 'user' : 'assistant', record.method + ' actual requestor');
}
for (let i = 1; i < records.length; i++) assert.ok(records[i].sequence > records[i - 1].sequence, 'receipt sequence increases');
const report = { schema: 'natlang.telecom-realhost-receipt-argument-preflight/1',
  codebase_revision: metadata.source_revision, source_id: fixtureRow.id,
  host_source_sha256: createHash('sha256').update(await readFile(root + '/' + binding.source_path)).digest('hex'),
  worker_sha256: createHash('sha256').update(await readFile(root + '/source/telecom_worker.py')).digest('hex'),
  fixture_sha256: createHash('sha256').update(await readFile(root + '/' + binding.fixtures_path)).digest('hex'),
  read_calls: records.length, selected_line_id: match[0].value.lineId,
  selected_line_matches_caller: true, writes: 0, records };
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, read_calls: report.read_calls, selected_line_id: report.selected_line_id, writes: 0 }));
