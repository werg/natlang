import { adaptedCase, requireValue, equal, sha, evalCall, returnCall } from './common.mjs';
import { nlFile } from '../inline-curriculum/lib.mjs';

/** Independent implementation of the curated deterministic source contracts. */
export function decisionGold(task) {
  const family = task.source_meta?.family_id, key = task.source_meta?.question_key;
  const text = task.state;
  if (family === 'smart_home_v2') {
    const m = /^Request: (on|off) the (\w+) in the (\w+)\. Authorization=(yes|no); people present=(\d+); time=\d+:\d+\.$/.exec(text);
    requireValue(m, 'unsupported_home_state');
    if (key === 'device') {
      const matches = Object.entries(task.criteria ?? {}).filter(([,v]) => v === `${m[3]} ${m[2]}`);
      requireValue(matches.length === 1, 'ambiguous_device'); return matches[0][0];
    }
    if (key === 'execute') return String(m[4] === 'yes' && (m[2] !== 'heater' || Number(m[5]) > 0));
    if (key === 'risk') return m[4] === 'no' ? '2' : m[2] === 'heater' && Number(m[5]) === 0 ? '1' : '0';
  }
  if (family === 'catalog_lookup_v2') {
    const m = /^Product record: item=[^;]+; color=([^;]+); size=([^;]+); status=(new|picked|packed|shipped); expedited=(yes|no)\.$/.exec(text);
    requireValue(m, 'unsupported_catalog_state');
    if (key === 'match') {
      const matches = Object.entries(task.criteria ?? {}).filter(([,v]) => v === `${m[1]}, ${m[2]}`);
      requireValue(matches.length === 1, 'ambiguous_catalog_match'); return matches[0][0];
    }
    if (key === 'ready') return String(m[3] === 'packed');
    if (key === 'stage') return String(['new','picked','packed','shipped'].indexOf(m[3]));
  }
  if (family === 'support_decisions_v1') {
    const m = /^工单主诉：([^。]+)。退款记录：([^。]+)。使用影响：([^。]+)。$/.exec(text);
    requireValue(m, 'unsupported_support_state');
    if (key === 'route') {
      const routes = {'忘记账户密码':'account', '软件按钮报错':'technical', '要求删除个人资料':'privacy', '包裹未送达':'shipping', '重复扣款':'billing', '询问操作手册位置':'documentation'};
      requireValue(routes[m[1]], 'unsupported_support_complaint'); return routes[m[1]];
    }
    if (key === 'completed') return String(m[2] === '退款已到账，处理完成');
    if (key === 'severity') {
      const matches = Object.entries(task.criteria ?? {}).filter(([,v]) => v.replace('没有','无') === m[3].replace('没有','无'));
      requireValue(matches.length === 1, 'ambiguous_support_severity'); return matches[0][0];
    }
  }
  throw new Error('no_independent_decision_oracle');
}

export function validateDecision(task) {
  requireValue(typeof task.state === 'string' && typeof task.instruction === 'string' && task.instruction.trim(), 'missing_decision_context');
  requireValue(Array.isArray(task.labels) && task.labels.length >= 2 && new Set(task.labels).size === task.labels.length, 'invalid_decision_labels');
  requireValue(task.labels.includes(task.gold), 'gold_not_in_labels');
  if (task.kind === 'boolean') requireValue(equal([...task.labels].sort(), ['false','true']), 'invalid_boolean_labels');
  if (task.kind !== 'boolean') requireValue(task.criteria && task.labels.every(label => typeof task.criteria[label] === 'string' && task.criteria[label].trim()), 'missing_label_criteria');
  if (task.gold_source !== 'programmatic') return {status:'held', checks:['unreviewed_teacher_or_synthetic_label']};
  const contractInstructions={
    'smart_home_v2:device':'Select exactly the room and device named in the request. Ignore authorization and occupancy for this question.',
    'smart_home_v2:execute':'Is execution allowed? Rule: authorization is required for every command; additionally any heater command requires at least one person present. Other devices do not require occupancy.',
    'smart_home_v2:risk':'Classify the control request using these mutually exclusive rules.',
    'catalog_lookup_v2:match':'Choose the candidate whose color AND size both match this item. Ignore item code and shipping status.',
    'catalog_lookup_v2:ready':'Is this item currently packed but not yet shipped? New, picked, and already shipped items do not qualify.',
    'catalog_lookup_v2:stage':'Select the stage explicitly reported in the record; expedited shipping does not change the current stage.',
    'support_decisions_v1:route':'根据记录，只按工单主诉选择支持团队；退款记录和功能影响是另外两个字段。',
    'support_decisions_v1:completed':'根据记录，退款是否已经实际到账完成？申请、处理中、批准或拒绝均不算到账。',
    'support_decisions_v1:severity':'根据记录，按明确记载的使用影响，选择最匹配的严重程度。',
  };
  requireValue(task.instruction===contractInstructions[`${task.source_meta?.family_id}:${task.source_meta?.question_key}`], 'unverified_decision_instruction');
  const fixedCriteria={
    'smart_home_v2:risk':{'0':'Authorized, and either not a heater or at least one person present','1':'Authorized heater request with zero people present','2':'Not authorized, regardless of device or occupancy'},
    'catalog_lookup_v2:stage':{'0':'New item, not picked','1':'Picked, not packed','2':'Packed, not shipped','3':'Already shipped'},
    'support_decisions_v1:severity':{'0':'功能正常，无使用障碍','1':'次要功能受阻，有替代办法','2':'核心功能受阻，有替代办法','3':'服务完全中断，无替代办法'},
  };
  const key=`${task.source_meta?.family_id}:${task.source_meta?.question_key}`;
  if(fixedCriteria[key])requireValue(task.labels.every(label=>task.criteria[label]===fixedCriteria[key][label]),'unverified_decision_criteria');
  if(key==='support_decisions_v1:route') {
    const meanings={account:'账户密码与身份验证',privacy:'个人资料删除与隐私请求',shipping:'包裹运输与投递问题',technical:'软件功能错误与故障',billing:'扣费、付款与退款问题',documentation:'查找说明文档与使用指南'};
    requireValue(task.labels.every(label=>task.criteria[label]===meanings[label]),'unverified_route_criteria');
  }
  requireValue(decisionGold(task) === task.gold, 'independent_decision_gold_mismatch');
  return {status:'eligible', checks:['independent_contract_oracle','label_domain_validated']};
}
const resultType = task => task.kind === 'boolean' ? 'boolean' : task.labels.map(JSON.stringify).join(' | ');
const result = task => task.kind === 'boolean' ? task.gold === 'true' : task.gold;
const instruction = task => `${task.instruction}\n${task.criteria ? `Label meanings: ${JSON.stringify(task.criteria)}\n` : ''}Return ${task.kind === 'boolean' ? 'a boolean' : 'exactly one of '+task.labels.map(JSON.stringify).join(', ')}.`;

export function decisionLeaf(task, info) {
  const quality = validateDecision(task);
  return adaptedCase(task, info, {family:'decision_leaf', suffix:'leaf',
    root:{name:'judge_task', args:{state:'string'}, returns:resultType(task), instructions:instruction(task)},
    inputs:{state:task.state}, expected:result(task), actions:[returnCall(result(task))],
    quality:quality.status, checks:quality.checks});
}

export function decisionBatch(tasks, info) {
  requireValue(tasks.length >= 2, 'batch_too_small');
  requireValue(new Set(tasks.map(t=>t.id)).size===tasks.length,'duplicate_batch_ids');
  const schema = task => sha([task.kind, task.labels, task.criteria, task.instruction]);
  requireValue(tasks.every(t=>schema(t)===schema(tasks[0])), 'mixed_decision_contracts');
  const qualities = tasks.map(validateDecision), type = resultType(tasks[0]);
  const visible = tasks.map(t=>({id:t.id, state:t.state})), values = tasks.map(result);
  const expected = {labels:Object.fromEntries(tasks.map((t,i)=>[t.id,values[i]])),
    counts:values.reduce((out,value)=>({...out,[String(value)]:(out[String(value)]??0)+1}),{})};
  const original = {...tasks[0], id:`batch:${sha(tasks.map(t=>t.id))}`, source_ids:tasks.map(t=>t.id),
    source_groups:tasks.map(t=>t.group_id), split:tasks.some(t=>t.split!=='train')?'test':'train'};
  const record = adaptedCase(original, info, {family:'decision_batch', suffix:'batch',
    root:{name:'review_tasks', args:{items:'DecisionTask[]'}, returns:'{ labels: Record<string, Label>, counts: Record<string, number> }',
      instructions:'Judge every item against judge_one’s stated rubric. Return its label by id and exact counts by label. Preserve item order when judging.'},
    files:{'types.ts':`export type DecisionTask = { id: string, state: string };\nexport type Label = ${type};\n`,
      'review_tasks/judge_one.nl':nlFile({args:{item:'DecisionTask'}, returns:'Label', instructions:instruction(tasks[0])+' The text to judge is item.state.'})},
    inputs:{items:visible}, expected,
    actions:[evalCall('const labels = await Promise.all(items.map(item => judge_one(item)));\nconst counts: Record<string, number> = {};\nfor (const label of labels) counts[String(label)] = (counts[String(label)] ?? 0) + 1;\nreturn {labels:Object.fromEntries(items.map((item,index)=>[item.id,labels[index]])), counts};'), returnCall(expected)],
    children:tasks.map(t=>({match:JSON.stringify(t.id), value:result(t)})),
    quality:qualities.every(q=>q.status==='eligible')?'eligible':'held', checks:[...new Set(qualities.flatMap(q=>q.checks))],
    sourceGroups:tasks.map(t=>`${info.source}:${t.group_id}`)});
  record.external_source.original_row = tasks;
  record.external_source.original_row_sha256 = sha(tasks);
  return record;
}
export const decisionSchema = task => sha([task.split, task.kind, task.labels, task.criteria, task.instruction]);
