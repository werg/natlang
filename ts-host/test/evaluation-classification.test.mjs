import test from 'node:test';
import assert from 'node:assert/strict';
import {classifyCaughtError} from '../scripts/evaluation-classification.mjs';
test('typed provider errors remain infrastructure failures',()=>{
 assert.equal(classifyCaughtError(Object.assign(new Error('sampler initialization'),{status:400})),'infrastructure_failure');
 assert.equal(classifyCaughtError(Object.assign(new TypeError('fetch failed'),{cause:{code:'UND_ERR_SOCKET'}})),'infrastructure_failure');
});
test('untyped text does not create infrastructure evidence',()=>{
 assert.equal(classifyCaughtError(new Error('HTTP 400 socket deadline')),'incomplete_task');
 assert.equal(classifyCaughtError(new Error('arbitrary'),{caseDeadlineExceeded:true}),'resource_failure');
});
