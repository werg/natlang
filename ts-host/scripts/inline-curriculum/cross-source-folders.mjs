// Independent operation-recipe and labeled-data selection. No labels in the visible workspace.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Random, curriculumCase, evalCall, returnCall, nlFile } from './lib.mjs';
import { labeledRows } from './folder-data.mjs';
import { SOURCES } from './acquire.mjs';
const bytes = readFileSync(new URL('../../../training/task-recipes/semantic-folder-operations.json',import.meta.url));
const recipes = JSON.parse(bytes).operations;
const recipeRevision = createHash('sha256').update(bytes).digest('hex');
const criteria = [
  { dataset:'sms_spam', labels:['spam'], text:'Select unsolicited promotional or fraudulent messages, excluding ordinary personal communications.' },
  { dataset:'sst2', labels:['positive'], text:'Select film reviews whose overall appraisal recommends or praises the film, considering qualifications and negation.' },
  { dataset:'sst2', labels:['negative'], text:'Select film reviews whose overall appraisal criticizes or discourages seeing the film, considering concessions and negation.' },
  { dataset:'banking77', labels:['card_payment_not_recognised','direct_debit_payment_not_recognised','cash_withdrawal_not_recognised','transaction_charged_twice','extra_charge_on_statement'],
    text:'Select complaints about unrecognized payments or withdrawals, duplicate charges, or unexpected extra charges. Exclude card delivery, activation, identity and account setup requests.' },
];
export function crossSourceFolders(seed,index,split='train') {
  const operation = recipes[new Random(seed,`operation:${index}`).int(0,recipes.length-1)];
  const spec = criteria[new Random(seed,`criterion:${index}`).int(0,criteria.length-1)];
  const rng = new Random(seed,`source-data:${index}`);
  const pool = labeledRows(spec.dataset,split).filter(row => spec.dataset !== 'sst2' || row.text.length>=80);
  const positives = pool.filter(row=>spec.labels.includes(row.label));
  const negatives = pool.filter(row=>!spec.labels.includes(row.label));
  const n = rng.int(8,18);
  const rows = rng.shuffle([...rng.sample(positives,rng.int(2,Math.floor(n/2))),...rng.sample(negatives,n)]).slice(0,n);
  if (!rows.some(row=>spec.labels.includes(row.label)) || !rows.some(row=>!spec.labels.includes(row.label))) throw new Error('need nontrivial source labels');
  const files={},ledger={},paths=[];
  for (const row of rows) {
    const group = `batch${rng.int(1,3)}`, id=row.id.slice(0,16), path=`groups/${group}/${id}.md`;
    files[path]=row.text+'\n';paths.push(path);ledger[id]=rng.int(1,1000);
  }
  const brief={ criterion:spec.text, operation:operation.id, instructions:operation.instruction };
  files['task.json']=JSON.stringify(brief,null,2)+'\n';files['ledger.json']=JSON.stringify(ledger,null,2)+'\n';
  const selected=rows.filter(row=>spec.labels.includes(row.label)).map(row=>row.id.slice(0,16)).sort();
  const total=selected.reduce((sum,id)=>sum+ledger[id],0);
  const expectedFiles={...files};
  let code=`const task = await folder.file('task.json').readJson();\nconst criterion = task.criterion;\nconst files = await folder.files('groups/**/*.md');\n`;
  const judge='await nl<boolean>`Decide whether document satisfies criterion. Read its contents and judge its meaning, including negation and exceptions.`(file)';
  if (operation.id==='iterated-selection') {
    code+=`type State = { cursor: number; ids: string[] };\nconst step = async (state: State): Promise<State> => {\n const file = files[state.cursor];\n const keep = ${judge};\n const id = file.path.split('/').pop().replace(/\\.md$/, '');\n return {cursor: state.cursor + 1, ids: keep ? [...state.ids, id] : state.ids};\n};\nconst final = await iterateOn(step, {cursor:0,ids:[]} as State).withMeasure(state => files.length - state.cursor).withLimit({maxSteps:32}).until(state => state.cursor >= files.length);\nconst selected = final.ids.sort();\n`;
  } else {
    code+=`const verdicts = await Promise.all(files.map(async file => ${judge}));\nconst chosen = files.filter((_, i) => verdicts[i]);\nconst selected = chosen.map(file => file.path.split('/').pop().replace(/\\.md$/, '')).sort();\n`;
  }
  if (operation.id==='weighted-total') {
    code+=`const weights = await folder.file('ledger.json').readJson();\nconst total = selected.reduce((sum,id) => sum + weights[id], 0);\nawait folder.file('total.txt').writeText(String(total)+'\\n');\nreturn total;`;
    expectedFiles['total.txt']=String(total)+'\n';
  } else if(operation.id==='route-documents') {
    code+=`for (const file of chosen) { const group = file.path.split('/')[1]; await file.moveTo('selected/'+group+'/'); }\nreturn selected;`;
    for(const path of paths) if(selected.includes(path.split('/').pop().replace(/\.md$/,''))) {expectedFiles[path.replace(/^groups\//,'selected/')]=files[path];delete expectedFiles[path];}
  } else {
    code+=`await folder.file('selection.json').writeText(JSON.stringify(selected));\nreturn selected;`;
    expectedFiles['selection.json']=JSON.stringify(selected);
  }
  const expected=operation.id==='weighted-total'?total:selected;
  const record=curriculumCase({family:'cross_source_folders',shape:`s${seed}-cross${index}`,variant:operation.id,
    slice:'inline_placement',domain:'other',mode:'single_call',inline:'required',...(operation.id==='iterated-selection'?{iterate:'required'}:{}),
    root:{name:'process_batches',kind:'directory-reducer',args:{},returns:operation.id==='weighted-total'?'number':'string[]',
      instructions:'Read task.json for the runtime-selected criterion and operation. Apply that operation to the documents in groups/. Use an inline natural-language function for each semantic document judgment, and code for exact file operations and aggregation. Available library helpers may solve only part of this request.'},
    files:{'process_batches/mentions_money.nl':nlFile({args:{document:'unknown'},returns:'boolean',description:'Whether the document mentions money, not whether it meets a task criterion.',instructions:'Read document and decide whether it mentions money.'})},
    folderFiles:files,expectedFiles,expected,split,
    minimumSequence:['read the selected operation and criterion','judge actual document contents in inline lambdas','apply exact operation and verify resulting files'],
    reference:{root:[evalCall(code),returnCall(expected)],children:[{match:'An iterative process',value:{verdict:'continue',reason:'The cursor advances by one toward the document count, retaining earlier selections.'}},...rows.map(row=>({match:row.id.slice(0,16),calls:[['read_file',{path:row.id.slice(0,16)+'.md'}],returnCall(spec.labels.includes(row.label))]}))]}});
  record.dataset=spec.dataset;record.dataset_records=rows.map(row=>row.id);
  record.source_ids=[...rows.map(row=>row.id),`operation:${operation.id}@${recipeRevision}`];
  record.source_groups=rows.map(row=>`${spec.dataset}:${row.id}`);
  record.source_revisions=[SOURCES[spec.dataset].revision,`semantic-folder-operations:${recipeRevision}`];
  record.license=SOURCES[spec.dataset].license;
  record.gold_sources=[`${spec.dataset}-labels`,'independent-operation-recipe','exact-file-transform'];
  record.generation.cross_product={version:1,operation:operation.id,recipe_sha256:recipeRevision,dataset:spec.dataset,files:rows.length,
    selection:'independent operation/criterion/data RNG streams; gold labels host-only'};
  return [record];
}
