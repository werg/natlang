import { definitionProject, lambdaSignature, programDefinition } from '../../dist/teacher/program.js';
import { modernType, modernSpec } from '../migrate-program-ir.mjs';
import { adaptedCase, requireValue, equal, sha, typeOf, evalCall, returnCall } from './common.mjs';

const instructions = text => String(text).replace(/`?(?:args|locals)\/(\w+)`?/g, '`$1`');
function modernFunction(spec) {
  const result=modernSpec(spec);
  if(result.instructions) result.instructions=instructions(result.instructions);
  if(result.codebase) result.codebase=Object.fromEntries(Object.entries(result.codebase).map(([name,child])=>[name,modernFunction(child)]));
  return result;
}

/** Preserve semantic programs as reviewable IR; only crisp source shapes admit offline. */
export function synthetic(original, info) {
  const sem=original.semantics, fold=sem.root?.$fold, old=sem.root?.$lambda ?? fold?.step?.$lambda;
  requireValue(old && ['lambda_source','lambda_graph','fold_graph','fold_source'].includes(original.kind), 'unsupported_synthetic_kind');
  const signature=lambdaSignature(modernType(old.type));
  const name=old.function ?? 'execute_task';
  const spec={...signature,instructions:instructions(old.instructions),types:Object.fromEntries(Object.entries(old.types??{}).map(([key,value])=>[key,modernType(value)])),
    codebase:Object.fromEntries(Object.entries(old.codebase??{}).map(([key,value])=>[key,modernFunction(value)]))};
  const foldType=fold && /^Fold<\s*([^,]+),\s*(.+)>$/.exec(modernType(fold.type));
  if(fold)requireValue(foldType,'unsupported_fold_type');
  const project=fold ? definitionProject('reduce_items',{args:{items:`${foldType[1]}[]`,initial:foldType[2]},returns:foldType[2],types:spec.types,
    instructions:'Starting at initial, apply step(acc, item) to each item in order. Pass the previous result as the next accumulator. Return the final accumulator. Preserve effects and state described by step.',codebase:{step:spec}}) : definitionProject(name,spec);
  let quality='held', checks=['legacy_types_and_scope_references_migrated','semantic_gold_requires_independent_review'];
  let actions=[returnCall(sem.expected)];
  if(sem.operation==='exact' && sem.formula) {
    const expressions={sum:'numbers.reduce((total,value)=>total+value,0)', count_over_100:'numbers.filter(value=>value>100).length',
      max:'Math.max(...numbers)',mean_round_2:'Math.round(numbers.reduce((total,value)=>total+value,0)/numbers.length*100)/100'};
    requireValue(expressions[sem.formula] && Array.isArray(sem.inputs.numbers) && sem.inputs.numbers.every(Number.isFinite), 'unsupported_crisp_formula');
    if(['max','mean_round_2'].includes(sem.formula)) requireValue(sem.inputs.numbers.length>0, 'empty_numeric_domain');
    const acceptedInstructions={
      sum:['Could you please add up the numbers in `args/numbers`?','Add up the numbers in `args/numbers`.','Sum the numbers in `args/numbers`.','- Add up the numbers in `args/numbers`','add up the numbers in `args/numbers`'],
      count_over_100:['Checklist item: How many numbers in `args/numbers` are above 100?','Count the numbers in `args/numbers` that are above 100.','Could you please tell me how many of the numbers in `args/numbers` are above 100?','how many nums in `args/numbers` over 100?','How many of the numbers in `args/numbers` are above 100?'],
      max:['Could you please tell me the largest number in `args/numbers`?','Find the largest number in `args/numbers`.','What is the largest number in `args/numbers`?','max number in `args/numbers`?','What is the max number in `args/numbers`?'],
      mean_round_2:['Could you please calculate the average of `args/numbers`, rounded to two decimals?','What is the average of `args/numbers`, rounded to two decimals?','- Calculate the average of `args/numbers`, rounded to two decimals.','Return the average of `args/numbers`, rounded to two decimals.','avg of `args/numbers` rounded to two decimals'],
    };
    requireValue(acceptedInstructions[sem.formula]?.includes(old.instructions),'unverified_crisp_instruction');
    requireValue(equal(signature.args,{numbers:'number[]'}) && signature.returns==='number','unverified_crisp_signature');
    // Independent host calculation; native replay checks the separately emitted expression.
    const numbers=sem.inputs.numbers;
    const computed=sem.formula==='count_over_100'?numbers.reduce((n,v)=>n+(v>100?1:0),0):sem.formula==='max'?numbers.slice().sort((a,b)=>b-a)[0]:
      sem.formula==='mean_round_2'?Math.round(numbers.reduce((a,b)=>a+b,0)*100/numbers.length)/100:numbers.reduce((a,b)=>a+b,0);
    requireValue(equal(computed,sem.expected),'crisp_synthetic_gold_mismatch');
    actions=[evalCall(`return ${expressions[sem.formula]};`),returnCall(sem.expected)]; quality='eligible';checks=['independent_crisp_formula_agreement'];
  }
  const root=fold ? {name:'reduce_items',args:{items:`${foldType[1]}[]`,initial:foldType[2]},returns:foldType[2],instructions:'Apply step in order and return the final accumulator.'} : {name,...spec};
  const record=adaptedCase(original,info,{family:'synthetic',suffix:original.family??'program',root,
    files:Object.fromEntries(Object.entries(project.files).filter(([path])=>path!==project.root)),
    inputs:fold ? {items:fold.over,initial:fold.init} : sem.inputs,expected:sem.expected,actions,quality,checks});
  // curriculumCase's root helper omits named type definitions; use the actual loader project.
  record.semantics.files[record.semantics.root]=project.files[project.root];
  programDefinition(record);
  record.external_source.original_row=original;
  record.legacy_reference={operations:sem.operations??null,leaf_oracles:sem.leaf_oracles??null,
    nested:sem.nested??null,review_required:quality!=='eligible'};
  return record;
}

/** Index candidate leaf definitions, refusing an ambiguous function-name join. */
export function leafDefinitions(originals) {
  const found=new Map();
  const add=(name,spec,types)=> {
    const candidate=modernFunction({...spec,types:{...types,...spec.types}});
    for(const [child,value] of Object.entries(spec.codebase??{})) add(child,value,{...types,...spec.types});
    if (!candidate.instructions || candidate.code) return;
    const identity=sha([candidate.args,candidate.returns,candidate.instructions,candidate.types]);
    const values=found.get(name)??new Map(); values.set(identity,candidate);found.set(name,values);
  };
  for (const original of originals) {
    const root=original.semantics?.root?.$lambda ?? original.semantics?.root?.$fold?.step?.$lambda;
    if(root) for (const [name,spec] of Object.entries(root.codebase??{})) add(name,spec,root.types??{});
  }
  return found;
}
export function reviewedLeaf(row,info,definitions) {
  requireValue(row.function && row.args && 'value' in row,'invalid_leaf_reference');
  const choices=[...(definitions.get(row.function)?.values()??[])].filter(spec=>
    equal(Object.keys(spec.args??{}).sort(),Object.keys(row.args).sort()));
  requireValue(choices.length===1,'ambiguous_or_missing_leaf_definition');
  const spec=choices[0], project=definitionProject('judge_reference',spec);
  const original={id:row.key,split:'train',source_ids:[row.key],source_groups:[`leaf-definition:${sha(spec)}`],license:'project-generated',gold_sources:['reviewed-legacy-leaf-bank']};
  const record=adaptedCase(original,info,{family:'leaf_reference',root:{name:'judge_reference',...spec},
    files:Object.fromEntries(Object.entries(project.files).filter(([name])=>name!==project.root)),inputs:row.args,expected:row.value,
    actions:[returnCall(row.value)],quality:'held',checks:['unique_definition_join','legacy_review_evidence_requires_current_semantic_review']});
  record.semantics.files[record.semantics.root]=project.files[project.root];
  programDefinition(record);
  record.external_source.original_row=row;
  return record;
}
