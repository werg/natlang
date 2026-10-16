import assert from 'node:assert/strict';
import { test } from 'node:test';
import { nativeQueryReadoutDependencies } from '../dist/native/native-derivative-contract.js';

test('decision-teacher readouts remain executable archive dependencies', () => {
  const a='nnt1_'+'a'.repeat(52), b='nnt1_'+'b'.repeat(52);
  const spec={kind:'native-query',terms:[{kind:'decision-distill',readouts:[a,null],teacher_readouts:[b,b]}]};
  assert.deepEqual(nativeQueryReadoutDependencies(spec),[a,b]);
  assert.throws(()=>nativeQueryReadoutDependencies({kind:'native-query',terms:[{teacher_readouts:'not-an-array'}]}),/exact tensor artifact/);
  assert.throws(()=>nativeQueryReadoutDependencies({kind:'native-query',terms:[{teacher_readouts:['nzs1_'+'a'.repeat(52)]}]}),/exact tensor artifact/);
});
