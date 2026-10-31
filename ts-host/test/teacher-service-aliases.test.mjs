import assert from 'node:assert/strict';
import { test } from 'node:test';
import { externalModule } from '../dist/native/external.js';
import { bindServiceAliases } from '../dist/teacher/service-aliases.js';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

test('service aliases expose separate declarations but share one stateful host adapter', async () => {
  const module = externalModule('telecom', `type Receipt<T> = { value: T };
    let sequence = 0, paidGb = 0;
    export async function accountRead(): Promise<Receipt<number>> { return { value: ++sequence }; }
    export async function paidAccountWrite(): Promise<number> { return ++sequence; }
    export async function refuelData(gb: number): Promise<number> {
      if (paidGb + gb > 2) throw new Error('case-wide paid GB cap exceeded');
      paidGb += gb; return paidGb;
    }
    export async function deviceCheck(): Promise<{sequence:number; paidGb:number}> {
      return { sequence: ++sequence, paidGb };
    }`);
  const services = { telecom: module.exports };
  const declarations = { telecom: module.declaration };
  bindServiceAliases(new Map([['telecom', module]]), {
    telecom: { source: 'telecom', methods: ['accountRead', 'paidAccountWrite', 'refuelData'] },
    device: { source: 'telecom', methods: ['deviceCheck'] },
  }, services, declarations);
  assert.deepEqual(Object.keys(services.telecom).sort(), ['accountRead', 'paidAccountWrite', 'refuelData']);
  assert.deepEqual(Object.keys(services.device), ['deviceCheck']);
  assert.match(declarations.telecom, /accountRead/);
  assert.match(declarations.telecom, /type Receipt<T>/, 'selected signatures retain their local type dependencies');
  assert.doesNotMatch(declarations.telecom, /deviceCheck/);
  assert.match(declarations.device, /deviceCheck/);
  assert.equal(await services.telecom.paidAccountWrite(), 1);
  assert.equal(await services.telecom.refuelData(1), 1);
  assert.deepEqual(await services.device.deviceCheck(), { sequence: 2, paidGb: 1 },
    'a device operation sees the same paid-consent budget as an account call');
  await assert.rejects(services.telecom.refuelData(2), /case-wide paid GB cap exceeded/,
    'cross-namespace operations cannot reset the cumulative paid-action allowance');
});

test('service aliases fail closed on missing exports or duplicate methods', () => {
  const module = externalModule('sim', 'export function read(): number { return 1; }');
  const services = { sim: module.exports }, declarations = { sim: module.declaration };
  assert.throws(() => bindServiceAliases(new Map([['sim', module]]), {
    device: { source: 'sim', methods: ['write'] },
  }, services, declarations), /unavailable method/);
  assert.throws(() => bindServiceAliases(new Map([['sim', module]]), {
    device: { source: 'sim', methods: ['read', 'read'] },
  }, services, declarations), /invalid service alias/);
});

test('an account-scoped callable cannot see or invoke the device namespace', async () => {
  const module = externalModule('sim', `export function accountRead(): number { return 1; }
    export function deviceWrite(): number { return 2; }`);
  const services = { sim: module.exports }, declarations = { sim: module.declaration };
  bindServiceAliases(new Map([['sim', module]]), {
    telecom: { source: 'sim', methods: ['accountRead'] }, device: { source: 'sim', methods: ['deviceWrite'] },
  }, services, declarations);
  const root = loadVirtualNatlang({
    'root.nl': '---\nargs: {}\nreturns: boolean\n---\nCall accountInspection and return true.',
    'root/accountInspection.nl': '---\nargs: {}\nreturns: boolean\n---\nTry a device call.',
  }, 'root.nl');
  const seen = {};
  const runtime = createNatlangRuntime({ services, serviceDeclarations: declarations,
    serviceScopes: { telecom: ['root/accountInspection.nl'], device: ['root/deviceInspection.nl'] },
    agent: async session => {
      const name = session.lam.functionName;
      seen[name] = Object.keys(session.availableServices()).sort();
      if (name === 'root') {
        const called = await session.applyAsync('eval', { code: 'await accountInspection(); return true;' });
        assert.equal(called.kind, 'ok', called.text); session.lam.return = true;
      } else {
        const attempted = await session.applyAsync('eval', { code: 'device.deviceWrite()' });
        assert.notEqual(attempted.kind, 'ok');
        assert.match(attempted.text, /device is not defined|Cannot find name 'device'/);
        session.lam.return = true;
      }
    } });
  assert.equal(await runtime.run(() => root()), true);
  assert.deepEqual(seen.root, []);
  assert.deepEqual(seen.accountInspection, ['telecom']);
});

test('readonly device alias excludes writes even when its source module has mutators', async () => {
  const module = externalModule('device', `export function checkStatusBar(): string { return 'airplane mode is on'; }
    export function turnAirplaneModeOff(): string { throw new Error('forbidden write reached'); }`);
  const services = { device: module.exports }, declarations = { device: module.declaration };
  bindServiceAliases(new Map([['device', module]]), {
    deviceRead: { source: 'device', methods: ['checkStatusBar'] },
  }, services, declarations);
  assert.deepEqual(Object.keys(services.deviceRead), ['checkStatusBar']);
  assert.equal(Object.hasOwn(services, 'device'), false);
  assert.match(declarations.deviceRead, /checkStatusBar/);
  assert.doesNotMatch(declarations.deviceRead, /turnAirplaneModeOff/);
  assert.equal(await services.deviceRead.checkStatusBar(), 'airplane mode is on');
  assert.equal(Object.hasOwn(services.deviceRead, 'turnAirplaneModeOff'), false,
    'the capability contains no callable write method');

  const root = loadVirtualNatlang({
    'root.nl': '---\nargs: {}\nreturns: boolean\n---\nCall the scoped inspection child.',
    'root/investigateDevice.nl': '---\nargs: {}\nreturns: boolean\n---\nTry an out-of-scope write.',
  }, 'root.nl');
  const runtime = createNatlangRuntime({ services, serviceDeclarations: declarations,
    serviceScopes: { deviceRead: ['root/investigateDevice.nl'] }, agent: async session => {
      if (session.lam.functionName === 'root') {
        const result = await session.applyAsync('eval', { code: 'await investigateDevice(); return true;', finish: true });
        assert.equal(result.kind, 'completed', result.text); session.lam.return = true;
      } else {
        const result = await session.applyAsync('eval', { code: 'await deviceRead.turnAirplaneModeOff(); return true;', finish: true });
        assert.notEqual(result.kind, 'ok');
        assert.match(result.text, /turnAirplaneModeOff|deviceRead/);
        session.lam.return = true;
      }
    } });
  assert.equal(await runtime.run(() => root()), true);
});
