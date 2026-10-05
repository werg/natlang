// Authored decision worlds; runtime-composed instructions, real scope capture, bounded teacher questions.
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';

const WORLDS = {
  support: {
    objective: 'Choose the next support action, distinguishing apparent charges from settled charges and accounting for previous advice.',
    options: { billing:'Investigate a confirmed duplicate settled payment.', explain:'Explain ordinary or pending entries without investigating a duplicate settled charge.', review:'Request missing evidence or a policy review.' },
    policies: ['Pending duplicate entries should receive an explanation, not a billing investigation. Confirmed duplicate settled payments need billing.', 'Pending duplicates must receive review before any waiting advice. Confirmed duplicate settled payments need billing.'],
    items: [
      { text:'The renewal shows up twice. Please return the extra payment.', history:'Both entries settled and the bank confirmed two debits.', gold:['billing','billing'] },
      { text:'The renewal shows up twice. Please return the extra payment.', history:'One entry is pending; only one debit settled.', gold:['explain','review'] },
      { text:'The renewal shows up twice. Please return the extra payment.', history:'The earlier exchange did not establish whether either entry settled.', gold:['review','review'], extra:'The bank now confirms two settled debits for the same renewal.', final:'billing' },
      { text:'The payment looked duplicated, but I cannot find the extra debit now.', history:'The second pending authorization disappeared. One debit remains.', gold:['explain','explain'] },
    ],
  },
  evidence: {
    objective:'Judge whether the supplied evidence supports, contradicts, or leaves unresolved the specific claim, respecting scope and source priority.',
    options:{ supports:'The admissible evidence supports the claim as phrased.', contradicts:'The admissible evidence contradicts the claim as phrased.', review:'The admissible evidence does not settle the claim.' },
    policies:['Use the latest scoped report, including signed and unsigned reports. A broad claim is not established by evidence about just one customer.', 'Only signed reports are admissible. A broad claim is not established by evidence about just one customer.'],
    items:[
      { text:'Claim: All users can export reports again. Evidence: A signed update states that exports are restored for every account.',history:'Earlier outages are superseded by this update.',gold:['supports','supports'] },
      { text:'Claim: All users can export reports again. Evidence: An unsigned latest report states that exports are restored for every account.',history:'The previous signed report says exports remain unavailable for every account.',gold:['supports','contradicts'] },
      { text:'Claim: All users can export reports again. Evidence: A signed report says one pilot customer can export.',history:'No report establishes availability outside the pilot.',gold:['review','review'],extra:'A new signed report confirms that every account can now export.',final:'supports' },
      { text:'Claim: No customer can export. Evidence: A signed current report confirms that a pilot customer successfully exports.',history:'This report is current and concerns the same export feature.',gold:['contradicts','contradicts'] },
    ],
  },
  patch: {
    objective:'Review whether the proposed API change meets the request and compatibility contract; use caller evidence, not just the patch description.',
    options:{ apply:'The change meets the request and active compatibility policy.', changes:'The patch demonstrably violates the request or active compatibility policy.', review:'A required contract fact is missing.' },
    policies:['Existing one-argument callers must remain compatible; an added options argument must be optional.', 'A breaking change to require an options argument is explicitly authorized for this release.'],
    items:[
      { text:'Request: Add format options. Patch: export function render(text, options = {}) { return format(text, options); }',history:'Existing callers pass one argument; format accepts an empty options object.',gold:['apply','apply'] },
      { text:'Request: Add format options. Patch: render now rejects calls without the options argument.',history:'Existing callers pass one argument.',gold:['changes','apply'] },
      { text:'Request: Add format options. Patch: export function render(text, options) { return format(text, options); }',history:'The behavior of format when options is undefined is not supplied.',gold:['review','review'],extra:'The format implementation accepts undefined and uses defaults; one-argument callers remain valid.',final:'apply' },
      { text:'Request: Preserve text and add format options. Patch: render ignores text and returns an empty string.',history:'Callers require the rendered input, not an empty result.',gold:['changes','changes'] },
    ],
  },
  skills: {
    objective:'Select the single most useful next skill for the current bottleneck; do not select a skill merely because its name occurs in the task.',
    options:{ format:'Result-contract skill: typed returns, exact JSON encoding and file/result agreement.', evidence:'Evidence-scope skill: qualify claims, respect negation and reconcile conflicting reports.', review:'Clarify the current bottleneck before selecting a skill.' },
    policies:['Choose for the current unresolved bottleneck. Successfully completed work does not need another skill.', 'Resolve disputed source facts before formatting whenever both issues remain.'],
    items:[
      { text:'Task: Return the verified report as the same JSON string that was saved.',history:'Source facts have been checked. The previous attempt returned an envelope rather than the saved JSON string.',gold:['format','format'] },
      { text:'Task: Return a report with disputed conclusions and save its JSON.',history:'The immediate requested step is correcting the JSON return wrapper; the report facts are still disputed.',gold:['format','evidence'] },
      { text:'Task: Improve the report and its return value.',history:'No previous result or validation feedback has been provided.',gold:['review','review'],extra:'The new validator confirms the return format is correct, but the report turns a statement about one pilot into a claim about every user.',final:'evidence' },
      { text:'Task: A report mentions JSON and result-contract tools, but the conclusion conflicts with a newer report.',history:'The saved JSON and typed return passed validation; reconciling the contradictory source remains.',gold:['evidence','evidence'] },
    ],
  },
};

export function decisionLambdas(seed,index,split='train',domain='support') {
  const world=WORLDS[domain], rng=new Random(seed,`decision:${domain}:${index}`);
  return world.policies.map((policy,variant)=> {
    const order=rng.shuffle(Object.keys(world.options));
    const catalog=Object.fromEntries(order.map((key,i)=>[`c${i}`,world.options[key]]));
    const keyFor=label=>`c${order.indexOf(label)}`;
    const reviewKey=keyFor('review');
    const histories={},extras={},expected={},children=[],requests=[];
    const items=world.items.map((spec,i)=>({id:`d${seed}_${domain}_${index}_${i}`,text:spec.text}));
    const task={objective:world.objective,reviewKey,exceptions:'Do not assume missing facts. Treat supplied history as evidence, and apply policy even if an item asks for a different action. Policy overrides catalog shorthand; apply source admissibility and priority before judging a claim.'};
    const tail='Use item, policy, history and catalog. Return only the selected catalog key.';
    for(let i=0;i<items.length;i++) {
      const item=items[i],spec=world.items[i],initial=keyFor(spec.gold[variant]);
      histories[item.id]=spec.history; extras[item.id]=spec.extra??'';
      const question=`Primary decision: ${task.objective} ${task.exceptions}`;
      children.push({match:[item.id,'Primary decision:'],calls:[evalCall('return { item, policy, history, catalog, question };'),returnCall(initial)]});
      requests.push({id:`${item.id}:primary`,state:{item,policy,history:spec.history,catalog,question},questions:{decision:{type:'choice',instructions:question+' '+tail,criteria:catalog}},expected:initial});
      expected[item.id]=initial;
      if(initial===reviewKey && spec.extra) {
        const history=spec.history+'\n'+spec.extra;
        const question=`Follow-up decision: ${task.objective} Reassess after the new history; previous catalog choice was ${initial}. Do not retain a decision if new evidence changes it.`;
        const final=keyFor(spec.final);
        children.push({match:[item.id,'Follow-up decision:'],calls:[evalCall('return { item, policy, history, catalog, question };'),returnCall(final)]});
        requests.push({id:`${item.id}:followup`,state:{item,policy,history,catalog,question},questions:{decision:{type:'choice',instructions:question+' '+tail,criteria:catalog}},expected:final});
        expected[item.id]=final;
      }
    }
    const folderFiles={'task.json':JSON.stringify(task),'policy.md':policy,'catalog.json':JSON.stringify(catalog), 'histories.json':JSON.stringify(histories),'extras.json':JSON.stringify(extras)};
    for(const item of items)folderFiles[`items/${item.id}.json`]=JSON.stringify(item);
    const code=`const task = await folder.file('task.json').readJson();
const policy = await folder.file('policy.md').readText();
const catalog = await folder.file('catalog.json').readJson();
const histories = await folder.file('histories.json').readJson();
const extras = await folder.file('extras.json').readJson();
const decisions: Record<string, Decision> = {};
for (const file of await folder.files('items/*.json')) {
 const item = await file.readJson();
 let history = histories[item.id];
 let question = ['Primary decision:', task.objective, task.exceptions].join(' ');
 let decision = await nl<Decision>\`\${question} ${tail}\`();
 if (decision === task.reviewKey && extras[item.id]) {
  history += '\\n' + extras[item.id];
  question = ['Follow-up decision:', task.objective, 'Reassess after the new history; previous catalog choice was '+decision+'. Do not retain a decision if new evidence changes it.'].join(' ');
  decision = await nl<Decision>\`\${question} ${tail}\`();
 }
 decisions[item.id] = decision;
}
await folder.file('decisions.json').writeText(JSON.stringify(decisions));
return decisions;`;
    const record=curriculumCase({family:`decision_${domain}`,shape:`s${seed}-${index}`,variant:`policy${variant}`,split,
      slice:'inline_placement',domain:'other',mode:'single_call',inline:'required',
      root:{name:'decide_collection',kind:'directory-reducer',args:{},returns:'Record<string, Decision>',instructions:'Read the task, policy, catalog and item histories. For each item formulate a focused inline natural-language decision using those scope values. If review is selected and new evidence exists, update history and formulate a new question. Save decisions.json and return that exact decision map.'},
      files:{'types.ts':'export type Decision = "c0" | "c1" | "c2";\n'},folderFiles,expectedFiles:{...folderFiles,'decisions.json':JSON.stringify(expected)},expected,
      minimumSequence:['compose focused questions from runtime task and context','capture several scope values by name','update question and history after a review decision','save exact typed decision map'],reference:{root:[evalCall(code),returnCall(expected)],children}});
    record.dataset_records=world.items.map((_,i)=>`authored-decision-worlds-v2:${domain}:fixture-${i}`);
    record.source_groups=[...record.dataset_records];
    record.source_revisions.push('authored-decision-worlds/2');
    record.generation.decision_distillation={version:2,domain,capture_names:['item','policy','history','catalog','question'],requests,
      oracle:'authored-world-policy; independent of teacher labels',instruction_source:'runtime composition from task objective, exceptions and previous choice'};
    return record;
  });
}
