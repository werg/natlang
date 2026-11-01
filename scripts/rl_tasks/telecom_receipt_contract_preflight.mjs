import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = resolve(process.argv[2]);
const output = resolve(process.argv[3]);
// This scripted contract check must not populate or contend for the production call store.
process.env.NATLANG_CALL_STORE = 'off';
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
const spool = resolve(process.env.NATLANG_TEACHER_JOURNAL_SPOOL_ROOT ?? dirname(output) + '/journal-spool');
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

// Exercise the authored receipt-consuming operation through the same Natlang loader and
// runtime used by the collector. This checks typed callable bindings and receipt
// return shapes with a scripted agent; actual model adherence to the NL instructions
// is a separate trajectory qualification.
const packageRoot = root + '/package';
const packageFiles = {};
async function collectFiles(dir, prefix = '') {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) await collectFiles(resolve(dir, entry.name), relative);
    else if (/\.(?:nl|ts|json|md)$/.test(entry.name)) packageFiles[relative] = await readFile(resolve(dir, entry.name), 'utf8');
  }
}
await collectFiles(packageRoot);
const tsHostDist = resolve(process.env.NATLANG_TS_HOST_DIST ?? new URL('../../ts-host/dist', import.meta.url).pathname);
const { nodePlatform } = await import(pathToFileURL(resolve(tsHostDist, 'runtime/node.js')).href);
const { loadVirtualCallables } = await import(pathToFileURL(resolve(tsHostDist, 'runtime/virtual-project.js')).href);
const { createNatlangRuntime } = await import(pathToFileURL(resolve(tsHostDist, 'runtime/runtime.js')).href);
const mounted = loadVirtualCallables(packageFiles, '', '/project', nodePlatform());
const accountReadService = Object.freeze({ lookupCustomerByPhone: service.lookupCustomerByPhone,
  getLineDetails: service.getLineDetails, getPlanDetails: service.getPlanDetails,
  getBillDetails: service.getBillDetails, getDataUsage: service.getDataUsage });
const scriptedAccountCode = `
  const lineReceipts = [];
  for (const lineId of customer.lineIds) {
    const receipt = await telecomRead.getLineDetails(lineId);
    if (receipt.arguments.id !== lineId) throw new Error("line receipt must bind requested ID through arguments.id");
    lineReceipts.push(receipt);
  }
  const digits = value => value.replace(/\\D/g, "");
  const matches = lineReceipts.filter(receipt => digits(receipt.value.phoneNumber) === digits(callerPhone));
  if (matches.length !== 1) return { lineId:null, line:null, plan:null, dataUsage:null, bills:[], observations:[] };
  const lineReceipt = matches[0];
  const planReceipt = await telecomRead.getPlanDetails(lineReceipt.value.planId);
  if (planReceipt.arguments.id !== lineReceipt.value.planId) throw new Error("plan receipt must bind requested ID through arguments.id");
  const usageReceipt = await telecomRead.getDataUsage(customer.customerId, lineReceipt.value.lineId);
  if (usageReceipt.arguments.customer_id !== customer.customerId || usageReceipt.arguments.line_id !== lineReceipt.value.lineId)
    throw new Error("usage receipt must bind customer_id and line_id");
  const observations = [];
  for (const receipt of [...lineReceipts, planReceipt, usageReceipt]) observations.push({
    receiptId:receipt.receiptId, sequence:receipt.sequence, source:receipt.toolName,
    value:receipt.value, capturedAtStep:0, supports:["account facts"]
  });
  return { lineId:lineReceipt.value.lineId, line:lineReceipt.value, plan:planReceipt.value,
    dataUsage:usageReceipt.value, bills:[], observations };
`;
const scriptedAgent = async session => {
  const functionName = String(session.lam.functionName);
  if (functionName.endsWith('lookupCustomer')) {
    const result = await session.applyAsync('eval', { code: `
      const receipt = await telecomRead.lookupCustomerByPhone(callerPhone);
      if (receipt.arguments.phone_number !== callerPhone) throw new Error("customer receipt must bind exact phone_number");
      return { customer:receipt.value, observation:{ receiptId:receipt.receiptId, sequence:receipt.sequence,
        source:receipt.toolName, value:receipt.value, capturedAtStep:0, supports:["account identity"] } };
    `, finish: true });
    assert.equal(result.kind, 'completed', result.text);
    return;
  }
  if (!functionName.endsWith('investigateAccount'))
    throw new Error(`unexpected callable during receipt preflight: ${functionName}`);
  const result = await session.applyAsync('eval', { code: scriptedAccountCode, finish: true });
  assert.equal(result.kind, 'completed', result.text);
};
const serviceDeclaration = `declare const telecomRead: {
  lookupCustomerByPhone(phoneNumber:string): Promise<{receiptId:string;sequence:number;toolName:string;requestor:'assistant'|'user';arguments:{phone_number:string};value:{customerId:string;fullName:string;phoneNumber:string;accountStatus:string;lineIds:string[];billIds:string[]}}>
  getLineDetails(lineId:string): Promise<{receiptId:string;sequence:number;toolName:string;arguments:{id:string};value:{lineId:string;phoneNumber:string;status:string;planId:string;roamingEnabled:boolean;contractEndDate:string|null;suspensionStartDate:string|null}}>
  getPlanDetails(planId:string): Promise<{receiptId:string;sequence:number;toolName:string;arguments:{id:string};value:{planId:string;name:string;dataLimitGB:number;dataRefuelingPricePerGB:number;pricePerMonth:number}}>
  getBillDetails(billId:string): Promise<{receiptId:string;sequence:number;toolName:string;arguments:{id:string};value:{billId:string;totalDue:number;dueDate:string;status:string}}>
  getDataUsage(customerId:string,lineId:string): Promise<{receiptId:string;sequence:number;toolName:string;arguments:{customer_id:string;line_id:string};value:{lineId:string;dataUsedGB:number;dataLimitGB:number;dataRefuelingGB:number;cycleEndDate:string}}>
}`;
const runtime = createNatlangRuntime({ platform: nodePlatform(), agent: scriptedAgent,
  services: { telecomRead: accountReadService }, serviceDeclarations: { telecomRead: serviceDeclaration },
  serviceScopes: { telecomRead: ['resolveIncident/advance/investigateAccount.nl',
    'resolveIncident/advance/lookupCustomer.nl'] },
  workspace: packageRoot, network: false, codeEdits: 'deny' });
const callerPhone = fixtureRow.fixture.request.callerPhone;
const lookup = await runtime.run(() => mounted.resolveIncident.advance.lookupCustomer(callerPhone));
assert.equal(lookup.customer.phoneNumber, customer.phoneNumber, 'authored lookup returns the exact host customer identity');
assert.ok(lookup.observation.receiptId && lookup.observation.sequence > 0 && lookup.observation.value.customerId,
  'authored lookup preserves its actual host receipt observation');
const account = await runtime.run(() => mounted.resolveIncident.advance.investigateAccount(
  lookup.customer, callerPhone,
  { caseRef: fixtureRow.fixture.request.caseRef, customerId: customer.customerId,
    lineId: null, customer, account: null, diagnosisState: null, diagnosisSummary: null,
    proposal: null, phase: 'investigating', status: 'in-progress', observations: [],
    receipts: [], attemptedActionIds: [], lastDiagnosis: '', remainingActions: 1,
    unresolvedEvidence: [], finalStatusBar: null, finalSpeedTest: null }));
assert.equal(account.lineId, match[0].value.lineId, 'authored account operation selects the exact caller line');
assert.equal(account.observations.length, customer.lineIds.length + 2,
  'authored account operation returns every line read plus exact plan and usage reads');
assert.ok(account.observations.every(row => typeof row.receiptId === 'string' && row.sequence > 0 && row.value),
  'authored account operation retains real host receipt identity and payloads');
const report = { schema: 'natlang.telecom-realhost-receipt-argument-preflight/1',
  codebase_revision: metadata.source_revision, source_id: fixtureRow.id,
  host_source_sha256: createHash('sha256').update(await readFile(root + '/' + binding.source_path)).digest('hex'),
  worker_sha256: createHash('sha256').update(await readFile(root + '/source/telecom_worker.py')).digest('hex'),
  fixture_sha256: createHash('sha256').update(await readFile(root + '/' + binding.fixtures_path)).digest('hex'),
  read_calls: records.length, selected_line_id: match[0].value.lineId,
  selected_line_matches_caller: true, writes: 0, records,
  authored_controller: { loaded_by: 'natlang.runtime.loadVirtualCallables', operation: 'resolveIncident/advance/investigateAccount.nl',
    runtime_dist: tsHostDist, returned_observations: account.observations.length,
    selected_line_id: account.lineId, returned_receipt_ids: account.observations.map(row => row.receiptId),
    lookup_operation: 'resolveIncident/advance/lookupCustomer.nl', lookup_receipt_id: lookup.observation.receiptId,
    actual_host_reads_exercised: true, scripted_model: true } };
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, read_calls: report.read_calls, selected_line_id: report.selected_line_id, writes: 0 }));
