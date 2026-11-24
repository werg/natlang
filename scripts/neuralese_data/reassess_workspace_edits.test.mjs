import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessWorkspaceEdit} from './reassess_workspace_edits.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const binding=value=>({complete:true,hash:sha(JSON.stringify(value)),bytes:Buffer.byteLength(JSON.stringify(value))});
function fixture(){
 const world={goal:'Correct the working notice.',files:{'notice.md':'Old statement.\n','scope.md':'Use approved facts.\n'},destination:'notice.md',source:'scope.md',concurrentAddition:'Desk note.\n',expected:'Correct statement.\nDesk note.\n'};
 const contents={...world.files,'notice.md':world.expected};
 const reads=Object.entries(contents).map(([path,text])=>({path,version:sha(text)}));
 const attempts=[{path:'notice.md',version:sha(world.files['notice.md']),text:world.expected,status:'changed'},{path:'notice.md',version:sha(world.files['notice.md']+world.concurrentAddition),text:world.expected,status:'written'}];
 const effects=[...reads.map(read=>({service:'goalWorkspace',method:'read',args:binding([read.path]),result:binding({...read,text:contents[read.path]})})),...attempts.map(attempt=>({service:'goalWorkspace',method:'revise',args:binding([attempt.path,attempt.version,attempt.text]),result:binding(attempt.status==='written'?{status:'written',version:sha(attempt.text)}:{status:attempt.status})}))];
 const report={status:'edited',path:'notice.md',version:sha(world.expected)};
 const packet={result:{passed:false},execution:{diagnostics:{openGoalVerification:{goal:world.goal,contents,reads,attempts,checks:{exactDocumentReadback:false,otherRequired:true}},natlangTeacher:{rootOutcome:'done',executionError:null,captureErrors:[],rootCallIds:['root'],entry:{arguments:{goal:world.goal}},report,calls:[{callId:'root',record:{output:binding(report),effects}}]}}}};
 const checker=()=>({checks:{exactDocumentReadback:true},edit:{retainedText:world.expected}});
 return {world,packet,checker};
}
test('reassessment retains the original failed result and all unrelated native checks',()=>{
 const {world,packet,checker}=fixture(),original=structuredClone(packet);
 assert.equal(reassessWorkspaceEdit(packet,world,checker).eligibleForContentAudit,true);
 assert.deepEqual(packet,original);
 packet.execution.diagnostics.openGoalVerification.checks.otherRequired=false;
 assert.equal(reassessWorkspaceEdit(packet,world,checker).eligibleForContentAudit,false);
});
test('reassessment rejects substituted root values, native read bytes and revision receipts',()=>{
 for(const tamper of [
  f=>f.packet.execution.diagnostics.natlangTeacher.report.version='invented',
  f=>f.packet.execution.diagnostics.openGoalVerification.reads[0].version='0'.repeat(64),
  f=>f.packet.execution.diagnostics.openGoalVerification.attempts[1].text='Unobserved write',
  f=>f.packet.execution.diagnostics.natlangTeacher.captureErrors.push('incomplete capture'),
 ]){const f=fixture();tamper(f);assert.throws(()=>reassessWorkspaceEdit(f.packet,f.world,f.checker));}
});
test('post-goal root failure remains telemetry when independent retained effects are established',()=>{
 const f=fixture(),native=f.packet.execution.diagnostics.natlangTeacher;
 native.rootOutcome='error';native.executionError={name:'PostGoalFailure',message:'Optional follow-up failed'};
 native.report=null;delete native.calls[0].record.output;
 assert.equal(reassessWorkspaceEdit(f.packet,f.world,f.checker).eligibleForContentAudit,true);
 assert.equal(native.rootOutcome,'error');assert.equal(native.executionError.message,'Optional follow-up failed');
});
