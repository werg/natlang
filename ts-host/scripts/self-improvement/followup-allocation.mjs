/** One finite research assignment owns every paid batch; declared maxima include existing development. */
import {AssignmentBudget,OperationJournal} from '../../dist/index.js';
import {join} from 'node:path';
export function followupAllocation(root){return new AssignmentBudget({collection:100,modelCalls:5000,caseExecutions:1800,trainingJobs:0,trainingUpdates:0,confirmation:36,cost:1000},{},1,new OperationJournal(join(root,'assignment')));}
export function enrollFollowup(root){const allocation=followupAllocation(root);allocation.allocateOnce('frozen-development',{collection:48,modelCalls:3600,caseExecutions:1200});allocation.allocateOnce('semantic-meta',{collection:1,modelCalls:400,caseExecutions:100});allocation.allocateOnce('final-confirmation',{collection:12,modelCalls:600,caseExecutions:100,confirmation:36});allocation.allocateOnce('recorded-action-replay',{collection:12,caseExecutions:300});return allocation.snapshot();}
