/** Exact action signatures feed the same type-guided generation as other natlang code. */
export const operationFields = {
  ide: {save:['target','text'],select:['target'],add:['target','text'],run:['text?'],inspect:['amount'],scenario:['text','secondary'],evaluate_case:['target']},
  notebook: {add:['target','secondary','text','ids?'],save:['target','text'],remove:['target'],execute:['target']},
  wiki: {save:['secondary','text'],add:['target','secondary','text'],select:['target'],incoming:['text'],merge:['text','ids?'],save_cell:['text'],run:['text?']},
  merge: {configure:['target','text','secondary'],right:['text'],propose:['text','ids?'],commit:[]},
  evidence: {add:['target','secondary','text'],search:['text'],claim:['target','text','secondary'],remove:['target']},
  publisher: {save:['target','secondary','text'],title:['text'],add:['secondary','text'],reorder:['ids'],publish:[]},
  types: {save:['text'],infer:['target','text','secondary'],diagnostic:['target','text'],clear:[]},
  economy: {price:['target','amount'],buy:['target','secondary','amount']},
  combat: {round:['text','secondary'],reset:[]},
  npc: {say:['text'],reply:['text','ids?','amount?'],remember:['text']},
  spells: {cast:['target','text','amount'],rest:[]},
  experiments: {hypothesis:['text'],run:['target','amount','count']},
  scheduling: {add:['text','amount'],date:['text'],schedule:['target','text'],complete:['target'],unschedule:['target']},
  logs: {ingest:['text'],incident:['text','target','ids'],resolve:['target'],escalate:['target']},
  data: {input:['text'],map:['text','secondary'],remove:['target'],transform:[]},
  workflows: {execute:['target'],fail:['target'],unknown:['target'],reconcile:['target','text'],compensate:['target']},
  tests: {save:['text','secondary'],add:['target','text','secondary'],run_case:['target']},
  media: {sample:[],import:['target'],transform:['target','text']},
  build: {save:['target','text'],build:[]},
  terminal: {execute:['text'],recipe:['target']},
  packages: {browse:[],resolve:['target','text'],install:['target']},
  repositories: {propose:['text','secondary'],check:[],reset:[]},
};
const types={amount:'Num',count:'Num',ids:'Text[]'};
export function decisionType(id) {
  const operations=operationFields[id];
  if(!operations)throw new Error(`Missing operation signatures for ${id}`);
  return 'export type Decision = '+Object.entries(operations).map(([action,fields])=>
    `{ action: ${JSON.stringify(action)}; ${fields.map(field=>`${field}: ${types[field.replace('?','')]??'Text'};`).join(' ')} }`).join(' | ')+';';
}
export function controlDecision(spec,event) {
  const fields=operationFields[spec.id][event.kind];
  if(!fields)throw new Error(`The ${event.kind} goal requires natlang orchestration`);
  const input=JSON.parse(event.value??'{}'),decision={action:event.kind};
  for(const field of fields){const key=field.replace('?','');if(Object.hasOwn(input,key))decision[key]=input[key];}
  return decision;
}
