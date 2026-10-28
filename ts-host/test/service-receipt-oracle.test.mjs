import assert from 'node:assert/strict';
import test from 'node:test';
import {checkRequiredServiceReceipt} from '../dist/teacher/service-receipt.js';

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const spec = {service: 'world', method: 'receipt'};

test('a successful literal requires the real read-only delivery receipt', async () => {
  let delivered = false;
  const services = {world: {receipt() { if (!delivered) throw Error('not delivered'); return 'delivered'; }}};
  assert.equal(await checkRequiredServiceReceipt(undefined, services, 'delivered', same), true);
  assert.equal(await checkRequiredServiceReceipt(spec, services, 'delivered', same), false);
  delivered = true;
  assert.equal(await checkRequiredServiceReceipt(spec, services, 'wrong', same), false);
  assert.equal(await checkRequiredServiceReceipt(spec, services, 'delivered', same), true);
  assert.equal(await checkRequiredServiceReceipt(spec, {}, 'delivered', same), false);
  await assert.rejects(checkRequiredServiceReceipt({service:'world', method:'drop'}, services, 'delivered', same));
});
