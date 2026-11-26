import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyNativeInvocationCertificate} from '../scripts/observed-workspace-evidence.mjs';
test('direct native verifier refuses a missing certificate before observing any artifact',async()=>{
 await assert.rejects(verifyNativeInvocationCertificate({call_id:'actual-finished-call'},new Map(),{},'/must-not-be-read'),/requires a source-bound certificate/);
});
