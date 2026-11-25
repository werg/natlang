import test from 'node:test';
import assert from 'node:assert/strict';
import {semanticGoalCompleted} from '../scripts/observed-goal-completion.mjs';

const fixture = () => ({
  interaction: {goal:'Correct the actual document',entry:{program:'edit.ts',callable:'edit',arguments:{}}},
  verification: {status:'passed',task_id:'task:open-goal-edit',checks:{actual:true},
    world:{context_id:'actual-context'},root_call_ids:['actual-root']},
  proof: {goal:'Correct the actual document',caseId:'edit',contextId:'actual-context',passed:true,
    entry:{program:'edit.ts',callable:'edit',arguments:{}},checks:{actual:true},
    outcomeDimensions:{effects:{passed:true},returnedValue:{passed:true,required:false}}},
});
test('only a whole bound semantic verdict permits execution-independent completion', () => {
  const {interaction,verification,proof}=fixture();
  const verdict=(p=proof,v=verification,e=[])=>semanticGoalCompleted(interaction,[p],v,e);
  assert.equal(verdict(),true);
  for(const changed of [{caseId:'sibling'},{contextId:'foreign'},{goal:'Different goal'},{entry:{}},
    {checks:{}},{checks:{actual:false}},{passed:false},
    {outcomeDimensions:{effects:{passed:false}}},
    {outcomeDimensions:{effects:{passed:true}},outcomeContract:{answer:{required:true}}},
    {outcomeDimensions:{effects:{passed:true,required:false}},outcomeContract:{effects:{required:true}}},
    {semanticReviewRequired:true}])assert.equal(verdict({...proof,...changed}),false,JSON.stringify(changed));
  assert.equal(verdict(proof,{...verification,status:'partial'}),false);
  assert.equal(verdict(proof,{...verification,checks:{}}),false);
  assert.equal(verdict(proof,{...verification,requirements:[]}),false);
  assert.equal(semanticGoalCompleted(interaction,[proof,proof],verification),false);
  assert.equal(semanticGoalCompleted(interaction,[],verification),false);
  assert.equal(verdict(proof,{...verification,requirements:{semantic_review:true}}),false);
  const review={schema:'natlang.live-workflow-review/1',model:'actual-model',input:{actual:'document'},
    verdict:{accepted:true},exchange:{request:{},reply:{}}};
  assert.equal(verdict({...proof,semanticReviewRequired:true},verification,[{semanticReview:review}]),true);
  assert.equal(verdict({...proof,semanticReviewRequired:true},verification,[{semanticReview:{...review,verdict:{accepted:false}}}]),false);
});
test('completed reassessment requires its verified source-bound proof',()=>{
  const {interaction,verification}=fixture();
  verification.reassessment={path:'assessment.json',sha256:'a'.repeat(64)};
  const proof={schema:'natlang.live-outcome-reassessment/1',task_id:verification.task_id,
    goal:interaction.goal,passed:true,checks:{actual:true}};
  assert.equal(semanticGoalCompleted(interaction,[proof],verification,[],false),false);
  assert.equal(semanticGoalCompleted(interaction,[proof],verification,[],true),true);
});
